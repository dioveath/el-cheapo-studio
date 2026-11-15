from pathlib import Path

import pytest

from studio.pipeline.motion import Motion
from studio.schemas import MotionSettings


def test_motion_reuse_preserves_generation_identity():
    settings = MotionSettings(prompt="A person walks forward.").model_dump()
    original = Motion(Path("/data"), settings)
    travelling = Motion(Path("/data"), settings | {"root_motion": "travel"})
    assert original.folder == travelling.folder
    for change in ({"seed": 43}, {"steps": 100}, {"duration": 10}, {"prompt": "A person jumps."}):
        changed = Motion(Path("/data"), settings | change)
        assert changed.folder != original.folder
        assert (changed.embedding == original.embedding) == ("prompt" not in change)


@pytest.mark.parametrize("values", [{"prompt": "  "}, {"duration": 11}, {"duration": 1}, {"steps": 10}, {"seed": -1}, {"root_motion": "unknown"}])
def test_invalid_motion_never_reaches_the_gpu(values):
    with pytest.raises(ValueError):
        MotionSettings.model_validate({"prompt": "A person walks."} | values)
