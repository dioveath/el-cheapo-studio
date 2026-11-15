import asyncio
from datetime import timedelta
import logging
import os
from pathlib import Path
import signal
from temporalio.client import Client
from temporalio.worker import Worker

from .activities import Activities, finish_job
from .workflows import AssetWorkflow, AnimationWorkflow, VideoWorkflow


async def main():
    Path("/tmp/worker-ready").unlink(missing_ok=True)
    logging.basicConfig(level=logging.INFO)
    client = await Client.connect(os.getenv("TEMPORAL_ADDRESS", "temporal:7233"))
    activities = Activities()
    worker = Worker(client, task_queue="low-poly-gpu", workflows=[AssetWorkflow, AnimationWorkflow, VideoWorkflow],
        activities=[activities.prepare_mesh, activities.rig_character, activities.generate_motion, activities.export_animation, activities.generate_video, finish_job],
        max_concurrent_activities=1, max_heartbeat_throttle_interval=timedelta(seconds=2),
        graceful_shutdown_timeout=timedelta(seconds=90))
    for interrupt in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(interrupt, lambda: asyncio.create_task(worker.shutdown()))
    Path("/tmp/worker-ready").touch()
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
