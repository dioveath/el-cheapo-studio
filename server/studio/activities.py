import asyncio
from contextlib import asynccontextmanager, suppress
import gc
import hashlib
import json
import time
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image
from psycopg.types.json import Jsonb
from temporalio import activity
from temporalio.exceptions import ApplicationError, CancelledError

from .db import DATA_ROOT, connection
from .pipeline.hunyuan import load_pipeline, MODEL_ID, MODEL_REVISION
from .pipeline.mesh import load_geometry, simplify_mesh, export_mesh, render_thumbnail
from .pipeline.performance import reconstruction_key, release_unused_cpu_memory
from .pipeline.rigging import Rigging, MODEL_REVISION as RIG_REVISION, SOURCE_REVISION as RIG_SOURCE
from .pipeline.motion import Motion


def update(job_id, phase, progress):
    with connection() as conn:
        row = conn.execute("UPDATE jobs SET phase=%s,progress=%s,updated_at=now() WHERE id=%s AND status NOT IN ('completed','failed','cancelled') RETURNING cancel_requested", (phase, int(progress), job_id)).fetchone()
    if not row or row["cancel_requested"] or activity.is_cancelled():
        raise CancelledError("Cancelled by user")


class Activities:
    def __init__(self):
        self._pipeline = None
        self._factory = None
        self._background = None

    async def _model(self, factory):
        import torch

        if self._factory != factory:
            self._pipeline = None
            self._factory = None
            gc.collect()
            torch.cuda.empty_cache()
            release_unused_cpu_memory()
            logging.info("GPU model release: %.3f GiB allocated", torch.cuda.memory_allocated() / 2**30)
            self._pipeline = await factory() if factory else None
            self._factory = factory
        return self._pipeline

    async def _compute(self, function, *args):
        task = asyncio.create_task(asyncio.to_thread(function, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            with suppress(Exception, asyncio.CancelledError):
                await task
            raise

    @asynccontextmanager
    async def _job(self, job_id):
        async def heartbeat():
            while True:
                activity.heartbeat()
                await asyncio.sleep(3)

        pulse = asyncio.create_task(heartbeat())
        try:
            with connection() as conn:
                job = conn.execute("SELECT * FROM jobs WHERE id=%s", (job_id,)).fetchone()
                if not job:
                    raise ValueError("Job no longer exists")
                if job["cancel_requested"]:
                    raise CancelledError("Cancelled by user")
                conn.execute("UPDATE jobs SET status='running',error=NULL,attempt=%s,updated_at=now() WHERE id=%s AND status!='completed'", (activity.info().attempt, job_id))
            yield job
        except (CancelledError, asyncio.CancelledError):
            with connection() as conn:
                conn.execute("UPDATE jobs SET status='cancelled',phase='Cancelled',updated_at=now() WHERE id=%s AND status!='completed'", (job_id,))
            raise
        except Exception as exc:
            with connection() as conn:
                conn.execute("UPDATE jobs SET status='retrying',phase='Preparing retry',error=%s,updated_at=now() WHERE id=%s AND status!='completed'", (str(exc)[:1000], job_id))
            if isinstance(exc, (ValueError, OSError)) or type(exc).__name__ == "OutOfMemoryError":
                message = "GPU memory ran out. Lower the decode batch size or stop other GPU apps." if type(exc).__name__ == "OutOfMemoryError" else str(exc)
                raise ApplicationError(message, non_retryable=True) from exc
            raise
        else:
            with connection() as conn:
                conn.execute("UPDATE jobs SET status='queued',updated_at=now() WHERE id=%s AND status='running'", (job_id,))
        finally:
            pulse.cancel()
            with suppress(asyncio.CancelledError):
                await pulse
            gc.collect()
            release_unused_cpu_memory()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    async def _reconstruct(self, job, asset):
        settings = job["settings"]
        job_id = str(job["id"])
        source = DATA_ROOT / asset["path"]
        image = None
        if asset["kind"] == "image":
            with Image.open(source) as opened:
                image = opened.convert("RGBA")
            image.thumbnail((2048, 2048))
            key = reconstruction_key(image, model=MODEL_ID, revision=MODEL_REVISION,
                steps=settings["steps"], resolution=settings["resolution"], seed=settings["seed"],
                remove_background=settings["remove_background"], decode_chunk_size=settings["decode_chunk_size"])
        else:
            key = hashlib.sha256(("mesh-import-v1:" + asset["sha256"]).encode()).hexdigest()
        with connection() as conn:
            stored = conn.execute("SELECT * FROM reconstructions WHERE key=%s", (key,)).fetchone()
        if stored and (DATA_ROOT / stored["original_path"]).is_file():
            update(job_id, "Reusing original shape", 75)
            mesh = await self._compute(load_geometry, DATA_ROOT / stored["original_path"])
            return mesh, key, True, stored["stats"]

        started = time.monotonic()
        gpu_stats = {}
        if image is None:
            update(job_id, "Reading mesh", 15)
            mesh = await self._compute(load_geometry, source)
        else:
            import torch
            update(job_id, "Preparing image", 5)
            if settings["remove_background"] and image.getextrema()[3][0] == 255:
                import rembg
                if self._background is None:
                    self._background = rembg.new_session(providers=["CPUExecutionProvider"])
                image = await self._compute(lambda: rembg.remove(image, session=self._background))
            if image.getextrema()[3][1] == 0:
                raise ValueError("No foreground remained. Try another image or disable background removal.")
            update(job_id, "Loading shape model", 8)
            pipeline = await self._model(load_pipeline)
            torch.cuda.reset_peak_memory_stats()

            def generate():
                def progress(step, timestep, outputs):
                    phase = "Building surface" if step + 1 == settings["steps"] else f"Generating shape · step {step + 1}/{settings['steps']}"
                    update(job_id, phase, 10 + 35 * (step + 1) / settings["steps"])

                with torch.inference_mode():
                    return pipeline(image=image, num_inference_steps=settings["steps"], guidance_scale=5.0,
                        generator=torch.Generator(device="cpu").manual_seed(settings["seed"]),
                        octree_resolution=settings["resolution"], num_chunks=settings["decode_chunk_size"],
                        mc_algo="mc", output_type="trimesh", enable_pbar=False,
                        callback=progress, callback_steps=1)[0]

            mesh = await self._compute(generate)
            torch.cuda.synchronize()
            gpu_stats = {"peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated()/2**30, 3),
                         "peak_cuda_reserved_gib": round(torch.cuda.max_memory_reserved()/2**30, 3)}
        if mesh is None or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise ValueError("No valid surface was generated. Try another image or seed.")
        update(job_id, "Saving original shape", 75)
        folder = DATA_ROOT / "originals" / key
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / f"{job_id}.glb"
        await self._compute(mesh.export, temporary)
        original_path = folder / "original.glb"
        temporary.replace(original_path)
        prepared_path = None
        if image is not None:
            image.save(folder / "input.png")
            prepared_path = str((folder / "input.png").relative_to(DATA_ROOT))
        stats = {"reconstruction_seconds": round(time.monotonic()-started, 2),
                 "original_vertices": len(mesh.vertices), "original_triangles": len(mesh.faces),
                 "model": MODEL_ID if image is not None else "Imported mesh", "cpu_offload": False, **gpu_stats}
        with connection() as conn:
            conn.execute("INSERT INTO reconstructions (key,original_path,prepared_path,stats) VALUES (%s,%s,%s,%s) ON CONFLICT(key) DO NOTHING",
                         (key, str(original_path.relative_to(DATA_ROOT)), prepared_path, Jsonb(stats)))
        return mesh, key, False, stats

    @activity.defn
    async def prepare_mesh(self, job_id: str) -> bool:
        async with self._job(job_id) as job:
            rig = job["operation"] == "rig" or job["settings"].get("rig", False)
            if all(job["artifacts"].get(key) and (DATA_ROOT / job["artifacts"][key]).is_file() for key in ("geometry", "glb", "obj", "thumbnail")):
                return rig
            if job["operation"] == "rig":
                raise ValueError("The saved mesh is missing. Choose another completed version.")
            started = time.monotonic()
            with connection() as conn:
                asset = conn.execute("SELECT * FROM assets WHERE id=%s", (job["asset_id"],)).fetchone()
            mesh, key, reused, original_stats = await self._reconstruct(job, asset)
            update(job_id, "Reducing polygons", 82)
            reduction_started = time.monotonic()
            mesh, simplification = await self._compute(simplify_mesh, mesh, job["settings"]["target_vertices"])
            reduction_seconds = time.monotonic() - reduction_started
            update(job_id, "Saving mesh", 95)
            folder = DATA_ROOT / "exports" / job_id
            folder.mkdir(parents=True, exist_ok=True)
            _, _, render_vertices = await self._compute(export_mesh, mesh, folder, job["settings"]["faceted"])
            await self._compute(mesh.export, folder / "geometry.glb")
            await self._compute(render_thumbnail, mesh, folder / "preview.png")
            stats = {**original_stats, "reconstruction_reused": reused,
                     "reconstruction_seconds": 0 if reused else original_stats["reconstruction_seconds"],
                     "vertices": len(mesh.vertices), "triangles": len(mesh.faces), "watertight": bool(mesh.is_watertight),
                     "render_vertices": render_vertices, "simplification": simplification,
                     "reduction_seconds": round(reduction_seconds, 2), "total_seconds": round(time.monotonic()-started, 2)}
            artifacts = {name: str(path.relative_to(DATA_ROOT)) for name, path in {
                "glb": folder / "mesh.glb", "obj": folder / "mesh.obj",
                "geometry": folder / "geometry.glb", "thumbnail": folder / "preview.png",
            }.items()}
            (folder / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
            with connection() as conn:
                conn.execute("UPDATE jobs SET reconstruction_key=%s,artifacts=%s,stats=%s,updated_at=now() WHERE id=%s",
                             (key, Jsonb(artifacts), Jsonb(stats), job_id))
            return rig

    async def _process(self, job_id, command, folder, gpu=False):
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "process.log").open("ab", buffering=0) as log:
            process = await asyncio.create_subprocess_exec(
                *command, env=os.environ if gpu else os.environ | {"CUDA_VISIBLE_DEVICES": ""},
                stdout=asyncio.subprocess.PIPE, stderr=log, start_new_session=True,
            )
            try:
                while line := await process.stdout.readline():
                    log.write(line)
                    if line.startswith(b"STUDIO "):
                        progress = json.loads(line[7:])
                        update(job_id, progress["phase"], progress["progress"])
                code = await process.wait()
                if code:
                    raise RuntimeError(f"Processing failed ({code}): " + (folder / "process.log").read_text(errors="replace")[-1200:])
            finally:
                if process.returncode is None:
                    import signal
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        await asyncio.wait_for(process.wait(), 5)
                    except TimeoutError:
                        os.killpg(process.pid, signal.SIGKILL)
                        await process.wait()

    @activity.defn
    async def rig_character(self, job_id: str):
        async with self._job(job_id) as job:
            if all(job["artifacts"].get(key) and (DATA_ROOT / job["artifacts"][key]).is_file() for key in ("rig-glb", "rig-fbx")):
                return
            started = time.monotonic()
            geometry = DATA_ROOT / job["artifacts"]["geometry"]
            signature = hashlib.sha256(geometry.read_bytes()).hexdigest()
            key = hashlib.sha256(json.dumps([signature, job["settings"]["faceted"], RIG_REVISION, RIG_SOURCE, 2]).encode()).hexdigest()
            folder = DATA_ROOT / "rigs" / key
            folder.mkdir(parents=True, exist_ok=True)
            reused = all((folder / name).is_file() for name in ("rig.json", "rig.glb", "rig.fbx"))
            if not reused:
                if not (folder / "prediction.npz").is_file():
                    update(job_id, "Loading rigging model", 5)
                    model = await self._model(Rigging.load)
                    mesh = await self._compute(load_geometry, geometry)
                    await self._compute(model.predict, mesh, folder, job["settings"]["faceted"], lambda phase, progress: update(job_id, phase, progress))
                update(job_id, "Saving character rig", 70)
                await self._process(job_id, ["/opt/blender/blender", "--background", "--python-use-system-env",
                    "--python-exit-code", "1", "--python", str(Path(__file__).parent / "pipeline/rigging.py"), "--", "export", str(folder)], folder)
            stats = json.loads((folder / "rig.json").read_text()) | {"reused": reused, "geometry_sha256": signature,
                        "total_seconds": round(time.monotonic() - started, 2)}
            artifacts = job["artifacts"] | {f"rig-{extension}": str((folder / f"rig.{extension}").relative_to(DATA_ROOT)) for extension in ("glb", "fbx")}
            with connection() as conn:
                conn.execute("UPDATE jobs SET artifacts=%s,stats=%s,updated_at=now() WHERE id=%s", (Jsonb(artifacts), Jsonb(job["stats"] | {"rigging": stats}), job_id))

    @activity.defn
    async def generate_motion(self, job_id: str):
        async with self._job(job_id) as job:
            motion = Motion(DATA_ROOT, job["motion"])
            reused = all((motion.folder / name).is_file() for name in ("motion.json", "motion.bvh", "motion.npz", "contacts.json"))
            if not reused:
                await self._model(None)
                folder = DATA_ROOT / "exports" / job_id / "animation"
                folder.mkdir(parents=True, exist_ok=True)
                settings = folder / "settings.json"
                settings.write_text(json.dumps(job["motion"]))
                command = ["/opt/kimodo/bin/python", "-I", str(Path(__file__).parent / "pipeline/motion.py")]
                if not motion.embedding.is_file():
                    update(job_id, "Encoding motion prompt", 15)
                    await self._process(job_id, command + ["encode", str(DATA_ROOT), str(settings)], folder, gpu=True)
                update(job_id, "Loading motion model", 30)
                await self._process(job_id, command + ["motion", str(DATA_ROOT), str(settings)], folder, gpu=True)
            stats = json.loads((motion.folder / "motion.json").read_text()) | {"reused": reused}
            artifacts = job["artifacts"] | {"motion-bvh": str((motion.folder / "motion.bvh").relative_to(DATA_ROOT)),
                                           "motion-data": str((motion.folder / "motion.npz").relative_to(DATA_ROOT))}
            with connection() as conn:
                conn.execute("UPDATE jobs SET artifacts=%s,stats=%s,updated_at=now() WHERE id=%s", (Jsonb(artifacts), Jsonb(job["stats"] | {"motion": stats}), job_id))

    @activity.defn
    async def generate_video(self, job_id: str):
        async with self._job(job_id) as job:
            folder = DATA_ROOT / "exports" / job_id / "video"
            folder.mkdir(parents=True, exist_ok=True)
            complete = all((folder / name).is_file() for name in ("video.json", "video.mp4", "preview.png"))
            if not complete:
                await self._model(None)
                source = None
                if job["asset_id"]:
                    with connection() as conn:
                        asset = conn.execute("SELECT path,kind FROM assets WHERE id=%s", (job["asset_id"],)).fetchone()
                    if not asset or asset["kind"] != "image":
                        raise ValueError("Video input must be an image.")
                    source = str(DATA_ROOT / asset["path"])
                settings = folder / "settings.json"
                settings.write_text(json.dumps(job["settings"] | {"image": source}))
                update(job_id, "Loading video model", 5)
                await self._process(job_id, ["/opt/ltx/bin/python", "-I", str(Path(__file__).parent / "pipeline/video.py"),
                    str(settings), str(folder)], folder, gpu=True)
            stats = json.loads((folder / "video.json").read_text())
            artifacts = {"mp4": str((folder / "video.mp4").relative_to(DATA_ROOT)),
                         "thumbnail": str((folder / "preview.png").relative_to(DATA_ROOT))}
            with connection() as conn:
                conn.execute("UPDATE jobs SET artifacts=%s,stats=%s,updated_at=now() WHERE id=%s",
                             (Jsonb(artifacts), Jsonb({"video": stats}), job_id))

    @activity.defn
    async def export_animation(self, job_id: str):
        async with self._job(job_id) as job:
            folder = DATA_ROOT / "exports" / job_id / "animation"
            folder.mkdir(parents=True, exist_ok=True)
            if not all((folder / name).is_file() for name in ("animation.json", "animation.glb", "animation.fbx")):
                update(job_id, "Retargeting character motion", 85)
                settings = folder / "settings.json"
                settings.write_text(json.dumps(job["motion"]))
                await self._process(job_id, ["/opt/blender/blender", "--background", "--python-use-system-env",
                    "--python-exit-code", "1", "--python", str(Path(__file__).parent / "pipeline/rigging.py"), "--", "animate",
                    str(DATA_ROOT / job["artifacts"]["rig-glb"]), str(DATA_ROOT / job["artifacts"]["motion-bvh"]), str(folder)], folder)
            stats = json.loads((folder / "animation.json").read_text())
            artifacts = job["artifacts"] | {f"animation-{extension}": str((folder / f"animation.{extension}").relative_to(DATA_ROOT)) for extension in ("glb", "fbx")}
            with connection() as conn:
                conn.execute("UPDATE jobs SET artifacts=%s,stats=%s,updated_at=now() WHERE id=%s", (Jsonb(artifacts), Jsonb(job["stats"] | {"animation": stats}), job_id))


@activity.defn
async def finish_job(data: dict):
    with connection() as conn:
        row = conn.execute("SELECT status,cancel_requested FROM jobs WHERE id=%s FOR UPDATE", (data["id"],)).fetchone()
        if row["status"] == "completed":
            return {"id": data["id"], "status": "completed"}
        status = "cancelled" if row["cancel_requested"] else data["status"]
        phase = {"completed": "Ready", "cancelled": "Cancelled", "failed": "Failed"}[status]
        conn.execute("UPDATE jobs SET status=%s,phase=%s,progress=CASE WHEN %s='completed' THEN 100 ELSE progress END,error=%s,updated_at=now() WHERE id=%s",
                     (status, phase, status, data.get("error") if status == "failed" else None, data["id"]))
    return {"id": data["id"], "status": status}
