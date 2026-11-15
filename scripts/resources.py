import asyncio
from datetime import timedelta
import json
import logging
import os
from pathlib import Path
import statistics
import time
from uuid import uuid4

from temporalio import activity, workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from concurrent.futures import ThreadPoolExecutor
    import gc
    import weakref
    import torch
    import trimesh
    from temporalio.client import Client
    from temporalio.worker import Worker
    from studio.pipeline import hunyuan
    from studio.pipeline.mesh import simplify_mesh


class Resources:
    def __init__(self, model):
        self.model = model
        self.events = []
        self.preparations = {}

    @activity.defn(name="resource_prepare")
    def prepare(self, request: dict) -> dict:
        started = time.perf_counter()
        model = self.model
        name = request["name"]
        self.preparations[name] = self.preparations.get(name, 0) + 1
        mesh = trimesh.load("/data/resource-probe/character.glb", force="mesh", process=False)
        mesh.merge_vertices()
        mesh, _ = simplify_mesh(mesh, max(100, len(mesh.vertices) // 2))
        path = Path("/data/resource-probe") / f"{name}.glb"
        mesh.export(path)
        result = {
            **request,
            "path": str(path),
            "vertices": len(mesh.vertices),
            "pid": os.getpid(),
            "model_id": id(model),
            "cuda_pointer": next(model.model.parameters()).data_ptr(),
            "allocated_mib": torch.cuda.memory_allocated() / 2**20,
        }
        self.events.append({"stage": "prepare", "name": name, "start": started, "end": time.perf_counter()})
        return result

    @activity.defn(name="resource_consume")
    def consume(self, artifact: dict) -> dict:
        started = time.perf_counter()
        if artifact.get("retry") and activity.info().attempt == 1:
            raise RuntimeError("Intentional downstream failure")
        model = self.model
        assert artifact["pid"] == os.getpid()
        assert artifact["model_id"] == id(model)
        assert artifact["cuda_pointer"] == next(model.model.parameters()).data_ptr()
        mesh = trimesh.load(artifact["path"], force="mesh", process=False)
        assert len(mesh.vertices) == artifact["vertices"]
        result = {
            **artifact,
            "attempt": activity.info().attempt,
            "allocated_after_mib": torch.cuda.memory_allocated() / 2**20,
        }
        self.events.append({"stage": "consume", "name": artifact["name"], "start": started, "end": time.perf_counter()})
        return result

    @activity.defn(name="resource_combined")
    def combined(self, request: dict) -> dict:
        return self.consume(self.prepare(request))


@workflow.defn
class ResourceWorkflow:
    @workflow.run
    async def run(self, request: dict) -> dict:
        options = {
            "start_to_close_timeout": timedelta(minutes=5),
            "retry_policy": RetryPolicy(initial_interval=timedelta(milliseconds=100), maximum_attempts=2),
        }
        if request["combined"]:
            return await workflow.execute_activity("resource_combined", request, **options)
        artifact = await workflow.execute_activity("resource_prepare", request, **options)
        return await workflow.execute_activity("resource_consume", artifact, **options)


async def main():
    logging.basicConfig(level=logging.ERROR)
    client = await Client.connect(os.getenv("TEMPORAL_ADDRESS", "temporal:7233"))
    queue = f"resource-probe-{uuid4().hex}"
    result = {
        "scope": "Actual Hunyuan weights and saved character geometry; no shape inference or MIA inference",
        "queue": queue,
        "device": torch.cuda.get_device_name(),
        "samples": [],
    }
    started = time.perf_counter()
    resources = Resources(await hunyuan.load_pipeline())
    result["model_startup_seconds"] = time.perf_counter() - started
    with ThreadPoolExecutor(max_workers=2) as executor:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[ResourceWorkflow],
            activities=[resources.prepare, resources.consume, resources.combined],
            activity_executor=executor,
            max_concurrent_activities=1,
        ):
            started = time.perf_counter()
            warmup = await client.execute_workflow(
                ResourceWorkflow.run, {"name": f"{queue}-warmup", "combined": False},
                id=f"{queue}-warmup", task_queue=queue,
            )
            result["warmup_workflow_seconds"] = time.perf_counter() - started
            result["model_allocated_mib"] = warmup["allocated_mib"]
            for index in range(8):
                for combined in ((True, False) if index % 2 == 0 else (False, True)):
                    name = f"{queue}-{index}-{combined}"
                    started = time.perf_counter()
                    sample = await client.execute_workflow(
                        ResourceWorkflow.run, {"name": name, "combined": combined},
                        id=name, task_queue=queue,
                    )
                    result["samples"].append({"combined": combined, "seconds": time.perf_counter() - started, **sample})
            retry_name = f"{queue}-retry"
            retry = await client.execute_workflow(
                ResourceWorkflow.run, {"name": retry_name, "combined": False, "retry": True},
                id=retry_name, task_queue=queue,
            )
            assert retry["attempt"] == 2
            assert resources.preparations[retry_name] == 1
            result["retry"] = {"downstream_attempts": retry["attempt"], "preparations": resources.preparations[retry_name]}
            parallel = await asyncio.gather(*[
                client.execute_workflow(
                    ResourceWorkflow.run, {"name": f"{queue}-parallel-{i}", "combined": False},
                    id=f"{queue}-parallel-{i}", task_queue=queue,
                ) for i in range(4)
            ])
            events = sorted(resources.events, key=lambda event: event["start"])
            assert all(previous["end"] <= current["start"] for previous, current in zip(events, events[1:]))
            assert len({sample["model_id"] for sample in parallel + result["samples"]}) == 1
            assert all(sample["allocated_mib"] == sample["allocated_after_mib"] for sample in parallel + result["samples"])
            result["concurrency"] = {"workflows": len(parallel), "overlapping_stages": 0}
            result["stage_order"] = [{"name": event["name"], "stage": event["stage"]} for event in events if "parallel" in event["name"]]
    for combined, label in ((True, "combined"), (False, "separate")):
        times = [sample["seconds"] for sample in result["samples"] if sample["combined"] == combined]
        result[label] = {"median_seconds": statistics.median(times), "min_seconds": min(times), "max_seconds": max(times)}
    reference = weakref.ref(resources.model)
    result["unload"] = {"before_mib": torch.cuda.memory_allocated() / 2**20}
    resources.model = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    assert reference() is None
    result["unload"]["after_mib"] = torch.cuda.memory_allocated() / 2**20
    started = time.perf_counter()
    resources.model = await hunyuan.load_pipeline()
    torch.cuda.synchronize()
    result["reload_seconds"] = time.perf_counter() - started
    result["unload"]["reloaded_mib"] = torch.cuda.memory_allocated() / 2**20
    result["model_shared_between_activities"] = True
    resources.model = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    result["final_allocated_mib"] = torch.cuda.memory_allocated() / 2**20
    Path("/data/resource-probe/results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("samples", "stage_order")}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
