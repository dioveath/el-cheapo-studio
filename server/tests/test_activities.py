import asyncio
import threading
import sys

import pytest

from studio.activities import Activities


def test_cancellation_waits_for_inflight_native_work():
    started = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    def native_work():
        started.set()
        release.wait(timeout=5)
        completed.set()

    async def run():
        task = asyncio.create_task(Activities()._compute(native_work))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            task.cancel()
            await asyncio.sleep(0.02)
            assert not task.done()
            assert not completed.is_set()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert completed.is_set()

    asyncio.run(run())


def test_rigging_metadata_is_independent_of_blender():
    import pymeshlab
    from model import PCAE
    from util.dataset_mixamo import KINEMATIC_TREE, BONES_IDX_DICT

    assert pymeshlab.MeshSet and PCAE
    assert len(KINEMATIC_TREE) == len(BONES_IDX_DICT) == 52
    assert KINEMATIC_TREE.parent_indices.count(-1) == 1
    assert "bpy" not in sys.modules


def test_subprocess_cancellation_reaps_the_gpu_owner(tmp_path):
    import os
    from pathlib import Path

    async def run():
        marker = tmp_path / "pid"
        code = "import os,time; from pathlib import Path; Path(" + repr(str(marker)) + ").write_text(str(os.getpid())); time.sleep(60)"
        task = asyncio.create_task(Activities()._process("test", [sys.executable, "-c", code], tmp_path))
        for _ in range(100):
            if marker.exists():
                break
            await asyncio.sleep(0.02)
        assert marker.exists()
        pid = int(marker.read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)

    asyncio.run(run())
