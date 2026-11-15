from uuid import uuid4

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from studio import api
from studio.schemas import CreateVideo, MotionSettings, VariantSettings, VideoSettings


def test_video_enforces_measured_memory_preset():
    request = CreateVideo(settings={"prompt": "  A red car drives forward.  "})
    assert request.asset_id is None
    assert request.settings.prompt == "A red car drives forward."
    assert request.settings.frames / request.settings.fps == 5.0625
    for override in ({"width": 1280}, {"frames": 161}, {"steps": 30}, {"cpu_offload": True}, {"prompt": "   "}, {"seed": -1}):
        with pytest.raises(ValidationError):
            VideoSettings.model_validate({"prompt": "A car drives forward."} | override)


def test_video_files_are_only_published_on_completion():
    row = {"id": uuid4(), "operation": "video", "status": "running", "artifacts": {
        "mp4": "exports/test/video/video.mp4", "thumbnail": "exports/test/video/preview.png",
        "internal": "exports/test/video/settings.json"}}
    assert api.serialize_job(row)["files"] == {}
    result = api.serialize_job(row | {"status": "completed"})
    assert set(result["files"]) == {"mp4"}
    assert result["preview_url"] == result["files"]["mp4"]
    assert result["original_url"] is None


@pytest.mark.parametrize("operation", ["variant", "rig", "animation"])
def test_video_cannot_enter_mesh_workflows(monkeypatch, operation):
    monkeypatch.setattr(api, "get_job", lambda _: {"status": "completed", "operation": "video"})
    with pytest.raises(HTTPException) as failure:
        if operation == "variant":
            api.create_variant(uuid4(), VariantSettings())
        elif operation == "rig":
            api.rig_character(uuid4())
        else:
            api.create_animation(uuid4(), MotionSettings(prompt="Walk forward"))
    assert failure.value.status_code == 409
