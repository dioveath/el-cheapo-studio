import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

OPTIONS = dict(
    start_to_close_timeout=timedelta(minutes=30),
    heartbeat_timeout=timedelta(seconds=90),
    cancellation_type=workflow.ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
    retry_policy=RetryPolicy(initial_interval=timedelta(seconds=5), maximum_attempts=3),
)


@asynccontextmanager
async def execution(job_id):
    result = {}
    try:
        yield result
        result.update(await workflow.execute_activity(
            "finish_job", {"id": job_id, "status": "completed"},
            start_to_close_timeout=timedelta(seconds=30),
        ))
    except asyncio.CancelledError:
        await asyncio.shield(workflow.execute_activity(
            "finish_job", {"id": job_id, "status": "cancelled", "error": None},
            start_to_close_timeout=timedelta(seconds=30),
        ))
        raise
    except Exception as exc:
        cause = exc
        while getattr(cause, "cause", None):
            cause = cause.cause
        await workflow.execute_activity(
            "finish_job", {"id": job_id, "status": "failed", "error": str(cause)[:1000]},
            start_to_close_timeout=timedelta(seconds=30),
        )
        raise


@workflow.defn
class AssetWorkflow:
    @workflow.run
    async def run(self, job_id: str) -> dict:
        async with execution(job_id) as result:
            rig = await workflow.execute_activity("prepare_mesh", job_id, **OPTIONS)
            if rig:
                await workflow.execute_activity("rig_character", job_id, **OPTIONS)
        return result


@workflow.defn
class AnimationWorkflow:
    @workflow.run
    async def run(self, job_id: str) -> dict:
        async with execution(job_id) as result:
            await workflow.execute_activity("rig_character", job_id, **OPTIONS)
            await workflow.execute_activity("generate_motion", job_id, **OPTIONS)
            await workflow.execute_activity("export_animation", job_id, **OPTIONS)
        return result


@workflow.defn
class VideoWorkflow:
    @workflow.run
    async def run(self, job_id: str) -> dict:
        async with execution(job_id) as result:
            await workflow.execute_activity("generate_video", job_id, **OPTIONS)
        return result
