from typing import Literal
from uuid import UUID
from pydantic import BaseModel, Field, model_validator


class VariantSettings(BaseModel):
    target_vertices: int = Field(1000, ge=0, le=200000)
    faceted: bool = True
    rig: bool = False
    @model_validator(mode="after")
    def vertex_budget(self):
        if self.target_vertices != 0 and self.target_vertices < 100:
            raise ValueError("Use 0 for the original, or a target of at least 100 vertices.")
        return self


class GenerationSettings(VariantSettings):
    steps: int = Field(30, ge=1, le=50)
    resolution: Literal[128, 256, 384] = 256
    seed: int = Field(42, ge=0, le=4294967295)
    remove_background: bool = True
    decode_chunk_size: Literal[1000, 2000, 4000, 8000] = 8000


class MotionSettings(BaseModel):
    model_config = {"str_strip_whitespace": True}
    prompt: str = Field(min_length=3, max_length=600)
    duration: int = Field(5, ge=2, le=10)
    steps: Literal[50, 100] = 50
    seed: int = Field(42, ge=0, le=4294967295)
    root_motion: Literal["in_place", "travel"] = "in_place"


class CreateJobs(BaseModel):
    asset_ids: list[UUID] = Field(min_length=1, max_length=16)
    settings: GenerationSettings = Field(default_factory=GenerationSettings)


class VideoSettings(BaseModel):
    model_config = {"str_strip_whitespace": True, "extra": "forbid"}
    prompt: str = Field(min_length=3, max_length=600)
    seed: int = Field(42, ge=0, le=4294967295)
    # Restrict generation to the preset measured on the 8 GB GPU.
    width: Literal[320] = 320
    height: Literal[192] = 192
    frames: Literal[81] = 81
    fps: Literal[16] = 16
    steps: Literal[8] = 8


class CreateVideo(BaseModel):
    asset_id: UUID | None = None
    settings: VideoSettings
