import asyncio
import gc
from .performance import CachedVolumeDecoder, release_unused_cpu_memory
MODEL_ID = "tencent/Hunyuan3D-2"
MODEL_REVISION = "9cd649ba6913f7a852e3286bad86bfa9a2d83dcf"
MODEL_SUBFOLDER = "hunyuan3d-dit-v2-0"

async def load_pipeline():
    import torch
    from huggingface_hub import hf_hub_download
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Start this service with NVIDIA GPU access.")

    names = ("config.yaml", "model.fp16.safetensors")
    downloads = await asyncio.gather(*[
        asyncio.to_thread(hf_hub_download, MODEL_ID, f"{MODEL_SUBFOLDER}/{name}", revision=MODEL_REVISION)
        for name in names
    ])
    paths = dict(zip(names, downloads))
    print(f"Loading {MODEL_ID}/{MODEL_SUBFOLDER} in FP16", flush=True)
    previous_dtype = torch.get_default_dtype()
    try:

        torch.set_default_dtype(torch.float16)
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
            paths["model.fp16.safetensors"],
            paths["config.yaml"],
            device="cuda",
            dtype=torch.float16,
            use_safetensors=True,
        )
    finally:
        torch.set_default_dtype(previous_dtype)

    for module in (pipeline.conditioner, pipeline.model, pipeline.vae):
        module.eval().requires_grad_(False)
    pipeline.vae.volume_decoder = CachedVolumeDecoder(pipeline.vae.volume_decoder)
    gc.collect()
    release_unused_cpu_memory()
    torch.cuda.empty_cache()
    print("Shape model ready; all weights remain on CUDA. CPU offload disabled.", flush=True)
    try:
        return pipeline
    finally:
        del pipeline, module
