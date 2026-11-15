"""LTX-Video's measured 8 GB preset, run in an isolated GPU subprocess."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.request

COMFY_REVISION = "eee59185ff4f43ae2d9ad122b245619d27d9daad"
GGUF_REVISION = "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"
WEIGHTS = (
    ("Lightricks/LTX-Video", "8984fa25007f376c1a299016d0957a37a2f797bb",
     "ltxv-2b-0.9.8-distilled-fp8.safetensors", "d6d8fa8ed3a98346787c2503ac80fb5d7cebcf80e356b79a2ba361fbadf97e15"),
    ("city96/t5-v1_1-xxl-encoder-gguf", "005a6ea51a7d0b84d677b3e633bb52a8c85a83d9",
     "t5-v1_1-xxl-encoder-Q3_K_S.gguf", "31ebe18e901bfb6e5709a20ec1c95fce29bce2b9545073231e0f909a53239f5c"),
)


def progress(phase, value):
    print("STUDIO " + json.dumps({"phase": phase, "progress": value}), flush=True)


def get_weights():
    cache = Path(os.getenv("HF_HOME", "/root/.cache/huggingface")) / "ltx-video"
    cache.mkdir(parents=True, exist_ok=True)
    paths = []
    for repo, revision, name, digest in WEIGHTS:
        path = cache / name
        if not path.is_file():
            progress("Downloading video models (first use)", 5)
            temporary = path.with_suffix(path.suffix + ".tmp")
            try:
                with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", timeout=120) as response, temporary.open("wb") as output:
                    while chunk := response.read(2**20):
                        output.write(chunk)
                with temporary.open("rb") as source:
                    if hashlib.file_digest(source, "sha256").hexdigest() != digest:
                        raise ValueError(f"Checksum mismatch for {name}.")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        paths.append(path)
    return paths


def install_row_embedding(ops, tensor_type, dequantize):
    """Dequantize only used vocabulary rows; every weight stays on CUDA."""
    import torch

    def forward(self, token_ids, out_dtype=None):
        weight = self.weight
        vocab, hidden = weight.tensor_shape
        if self.max_norm is not None or weight.patches:
            raise ValueError("The video preset requires an unpatched T5 embedding.")
        unique, inverse = torch.unique(token_ids.reshape(-1), return_inverse=True)
        packed = weight.as_subclass(torch.Tensor).reshape(vocab, -1)
        selected = tensor_type(packed.index_select(0, unique), tensor_type=weight.tensor_type,
                               tensor_shape=torch.Size((len(unique), hidden)), patches=[])
        decoded = dequantize(selected, dtype=out_dtype, dequant_dtype=self.dequant_dtype)
        return decoded.as_subclass(torch.Tensor).index_select(0, inverse).reshape(*token_ids.shape, hidden)

    ops.Embedding.forward_ggml_cast_weights = forward


def generate(settings, folder):
    started = time.monotonic()
    checkpoint, encoder = get_weights()
    # Configure before importing any module that selects devices or allocators.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, "/opt/comfyui")
    from comfy.cli_args import args
    args.gpu_only = True
    args.disable_dynamic_vram = True
    args.disable_async_offload = True
    args.disable_pinned_memory = True
    args.disable_fast_disk = True
    args.use_pytorch_cross_attention = True
    args.bf16_vae = True
    args.bf16_text_enc = True
    args.fp8_e4m3fn_unet = True

    import av
    import numpy as np
    import pynvml
    import torch
    from PIL import Image, ImageOps
    import folder_paths
    import comfy.sd
    import comfy.sample
    import comfy.samplers
    import comfy.model_management as mm

    torch.set_num_threads(4)
    torch.manual_seed(settings["seed"])
    # Memory pressure must fail the job, never evict weights to the CPU.
    mm.free_memory = lambda *a, **kw: []
    package = Path("/opt/comfyui/custom_nodes/ComfyUI-GGUF")
    spec = importlib.util.spec_from_file_location("gguf_nodes", package / "__init__.py",
                                                submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from gguf_nodes.nodes import CLIPLoaderGGUF
    from gguf_nodes.ops import GGMLOps, GGMLTensor
    from gguf_nodes.dequant import dequantize_tensor

    install_row_embedding(GGMLOps, GGMLTensor, dequantize_tensor)
    folder_paths.add_model_folder_path("text_encoders", str(encoder.parent))
    folder_paths.add_model_folder_path("clip", str(encoder.parent))
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    peak = [pynvml.nvmlDeviceGetMemoryInfo(handle).used]
    stopped = threading.Event()

    def monitor():
        while not stopped.wait(0.1):
            peak[0] = max(peak[0], pynvml.nvmlDeviceGetMemoryInfo(handle).used)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    checks = []

    def audit(stage):
        torch.cuda.synchronize()
        devices = {name: sorted({str(p.device) for p in model.parameters() if p.numel()})
                   for name, model in models.items()}
        if any(value != ["cuda:0"] for value in devices.values()):
            raise RuntimeError("All video model weights must remain on the GPU.")
        checks.append({"stage": stage, "parameter_devices": devices})
        peak[0] = max(peak[0], pynvml.nvmlDeviceGetMemoryInfo(handle).used)

    try:
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            progress("Loading video model", 10)
            model, _, vae, _ = comfy.sd.load_checkpoint_guess_config(
                str(checkpoint), output_clip=False, output_vae=True,
                model_options={"dtype": torch.float8_e4m3fn}, disable_dynamic=True)
            clip, = CLIPLoaderGGUF().load_clip(encoder.name, type="ltxv")
            mm.load_models_gpu([model, clip.patcher, vae.patcher], force_full_load=True)
            models = {"transformer": model.model, "text_encoder": clip.cond_stage_model,
                      "vae": vae.first_stage_model}
            audit("loaded")
            progress("Encoding video prompt", 20)
            positive = clip.encode_from_tokens_scheduled(clip.tokenize(settings["prompt"]))
            for _, metadata in positive:
                metadata["frame_rate"] = float(settings["fps"])
            negative = [[torch.zeros_like(c), dict(m)] for c, m in positive]
            audit("prompt")
            width, height, frames = settings["width"], settings["height"], settings["frames"]
            shape = (1, 128, (frames - 1) // 8 + 1, height // 32, width // 32)
            latent = torch.zeros(shape, device="cuda", dtype=torch.float32)
            mask = None
            if settings.get("image"):
                progress("Encoding starting image", 25)
                with Image.open(settings["image"]) as opened:
                    rgba = ImageOps.exif_transpose(opened).convert("RGBA")
                    background = Image.new("RGBA", rgba.size, "white")
                    background.alpha_composite(rgba)
                    image = ImageOps.fit(background.convert("RGB"), (width, height), method=Image.Resampling.BILINEAR)
                pixels = torch.from_numpy(np.asarray(image).copy()).to(device="cuda", dtype=torch.float32)[None] / 255
                encoded = vae.encode(pixels)
                latent[:, :, :encoded.shape[2]] = encoded
                mask = torch.ones((1, 1, latent.shape[2], 1, 1), device="cuda")
                mask[:, :, :encoded.shape[2]] = 0
                del pixels, encoded
                audit("image")
            noise = torch.randn(shape, generator=torch.Generator(device="cuda").manual_seed(settings["seed"]), device="cuda")
            sigmas = torch.tensor([1.0, .99375, .9875, .98125, .975, .909375, .725, .421875, 0.0])

            def callback(step, x0, x, total):
                progress(f"Generating video · step {step + 1}/{total}", 30 + 45 * (step + 1) / total)

            samples = comfy.sample.sample_custom(model, noise, 1.0, comfy.samplers.sampler_object("euler"),
                sigmas, positive, negative, latent, noise_mask=mask, callback=callback, seed=settings["seed"])
            audit("sampled")
            progress("Decoding video", 80)
            images = vae.decode_tiled(samples, tile_x=4, tile_y=4, overlap=1, tile_t=2, overlap_t=1)
            if images.ndim == 5:
                images = images.reshape(-1, *images.shape[-3:])
            audit("decoded")
            if tuple(images.shape) != (frames, height, width, 3) or not bool(torch.isfinite(images).all()):
                raise ValueError("The video model returned invalid frames.")
            pixels = (images.clamp(0, 1).float().cpu().numpy() * 255).round().astype(np.uint8)
            progress("Saving video", 95)
            temporary = folder / "video.tmp.mp4"
            with av.open(str(temporary), "w") as container:
                stream = container.add_stream("libx264", rate=settings["fps"])
                stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
                stream.options = {"crf": "20", "preset": "fast"}
                for frame in pixels:
                    for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            temporary.replace(folder / "video.mp4")
            Image.fromarray(pixels[frames // 2]).save(folder / "preview.png")
            stats = {"model": "LTX-Video-2B-0.9.8-distilled-FP8", "text_encoder": "T5-XXL-Q3_K_S",
                "comfy_revision": COMFY_REVISION, "gguf_revision": GGUF_REVISION,
                "weight_sha256": {entry[2]: entry[3] for entry in WEIGHTS},
                "cpu_offload": False, "embedding_rows_only": True, "device_checks": checks,
                "width": width, "height": height, "frames": frames, "fps": settings["fps"],
                "duration": frames / settings["fps"], "steps": settings["steps"],
                "mode": "image_to_video" if settings.get("image") else "text_to_video",
                "generation_seconds": round(time.monotonic() - started, 2),
                "peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                "peak_total_vram_gib": round(peak[0] / 2**30, 3)}
            temporary = folder / "video.tmp.json"
            temporary.write_text(json.dumps(stats, indent=2))
            temporary.replace(folder / "video.json")
    finally:
        stopped.set()
        thread.join(timeout=1)
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("settings", type=Path)
    parser.add_argument("folder", type=Path)
    options = parser.parse_args()
    sys.argv = [sys.argv[0]]
    generate(json.loads(options.settings.read_text()), options.folder)
