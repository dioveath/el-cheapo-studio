import argparse
from contextlib import contextmanager
import gc
import importlib.metadata
import json
from pathlib import Path
import resource
import threading
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
from studio.pipeline.motion import MODELS, SOURCE_REVISION

import numpy as np
import torch
from huggingface_hub import snapshot_download

PROMPTS = {
    "walk": "A person walks forward at a relaxed, steady pace.",
    "jump": "A person jumps straight up with both feet and lands on both feet.",
    "wave": "A person stands in place and waves hello with their right hand.",
}
parser = argparse.ArgumentParser()
parser.add_argument("stage", choices=("download", "encode", "motion"))
parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts/kimodo")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
if args.stage == "download":
    for repo, revision in MODELS.values():
        print(snapshot_download(repo, revision=revision, max_workers=2), flush=True)
    raise SystemExit(0)
torch.set_num_threads(8)
assert torch.cuda.is_available()
torch.cuda.init()
report = {
    "stage": args.stage,
    "gpu": torch.cuda.get_device_name(),
    "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
    "cpu_offload": False,
    "source_revision": SOURCE_REVISION,
    "models": MODELS,
    "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "bitsandbytes", "peft", "accelerate", "kimodo")},
    "measurements": [],
}


@contextmanager
def measurement(name):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    sample = {"name": name, "device_used_peak_gib": 0}
    stopped = threading.Event()

    def poll():
        while not stopped.is_set():
            free, total = torch.cuda.mem_get_info()
            sample["device_used_peak_gib"] = max(sample["device_used_peak_gib"], (total - free) / 2**30)
            stopped.wait(0.02)

    monitor = threading.Thread(target=poll, daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        yield sample
        torch.cuda.synchronize()
        sample["status"] = "passed"
    except Exception as error:
        sample["status"] = "failed"
        sample["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        sample["seconds"] = time.perf_counter() - started
        stopped.set()
        monitor.join()
        sample["allocated_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        sample["reserved_peak_gib"] = torch.cuda.max_memory_reserved() / 2**30
        sample["process_peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        report["measurements"].append(sample)
        (args.output / f"{args.stage}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(sample), flush=True)


if args.stage == "encode":
    from kimodo.model.llm2vec.llm2vec import LLM2Vec
    from kimodo.sanitize import sanitize_texts

    repo, revision = MODELS["encoder"]
    path = snapshot_download(repo, revision=revision, local_files_only=True)
    with measurement("encoder_load") as sample:
        encoder = LLM2Vec.from_pretrained(
            base_model_name_or_path=path,
            peft_model_name_or_path=str(Path(path) / "supervised_adapter"),
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            low_cpu_mem_usage=True,
        ).eval().requires_grad_(False)
        encoder.model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        devices = sorted({str(parameter.device) for parameter in encoder.parameters()})
        assert devices == ["cuda:0"], devices
        sample["parameter_devices"] = devices
        sample["quantization"] = encoder.model.config.quantization_config.to_dict()
    embeddings = {}
    for repeat in range(3):
        for name, prompt in PROMPTS.items():
            text = sanitize_texts([prompt])[0]
            with measurement(f"encode_{name}_{repeat}") as sample, torch.inference_mode():
                sentence = encoder._convert_to_str("", text)
                features = encoder.tokenize([encoder.prepare_for_tokenization(sentence)])
                features = {key: value.to("cuda") for key, value in features.items()}
                vector = encoder(features).float().cpu().numpy()[0]
                assert vector.shape == (4096,) and np.isfinite(vector).all()
                sample["prompt"] = text
                sample["tokens"] = features["input_ids"].shape[1]
                if name in embeddings:
                    sample["maximum_repeat_error"] = float(np.abs(vector - embeddings[name]).max())
                embeddings[name] = vector
    np.savez_compressed(args.output / "embeddings.npz", **embeddings)
    del encoder, features
    gc.collect()
    torch.cuda.empty_cache()
    report["allocated_after_unload_gib"] = torch.cuda.memory_allocated() / 2**30
    assert report["allocated_after_unload_gib"] < 0.05
else:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from kimodo.exports.bvh import save_motion_bvh
    from kimodo.exports.motion_io import save_kimodo_npz
    from kimodo.sanitize import sanitize_texts
    from kimodo.tools import seed_everything

    embeddings = dict(np.load(args.output / "embeddings.npz"))
    texts = {sanitize_texts([prompt])[0]: torch.from_numpy(embeddings[name]) for name, prompt in PROMPTS.items()}
    repo, revision = MODELS["motion"]
    path = snapshot_download(repo, revision=revision, local_files_only=True)
    with measurement("motion_load") as sample:
        config = OmegaConf.load(Path(path) / "config.yaml")
        config.checkpoint_dir = path
        config = OmegaConf.to_container(config, resolve=True)
        config.pop("checkpoint_dir")
        model = instantiate(config, text_encoder=None, device="cuda").eval().requires_grad_(False)
        model.text_encoder = lambda prompts: (torch.stack([texts[text] for text in prompts])[:, None].cuda(), [1] * len(prompts))
        devices = sorted({str(parameter.device) for parameter in model.parameters()})
        assert devices == ["cuda:0"], devices
        sample["parameter_devices"] = devices
        sample["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    skeleton = model.output_skeleton
    (args.output / "skeleton.json").write_text(json.dumps({
        "names": skeleton.bone_order_names,
        "parents": skeleton.joint_parents.tolist(),
        "rest_positions": skeleton.neutral_joints.tolist(),
        "fps": model.fps,
    }, indent=2) + "\n")
    cases = [("warmup", "walk", 3, 50, 0)]
    cases += [(f"walk-{duration}s-{steps}-{seed}", "walk", duration, steps, seed) for duration in (3, 5, 10) for steps in (50, 100) for seed in (42, 43, 44)]
    cases += [(f"{name}-5s-100-{seed}", name, 5, 100, seed) for name in ("jump", "wave") for seed in (42, 43, 44)]
    for name, motion, duration, steps, seed in cases:
        with measurement(name) as sample:
            seed_everything(seed)
            sample.update(prompt=PROMPTS[motion], duration=duration, steps=steps, seed=seed, post_processing=True)
            with torch.inference_mode():
                output = model([PROMPTS[motion]], [int(duration * model.fps)], num_denoising_steps=steps,
                               post_processing=True, return_numpy=False, progress_bar=iter)
                output = {key: value[0] for key, value in output.items()}
            torch.cuda.synchronize()
        with measurement(f"export_{name}") as sample:
            data = {key: value.detach().cpu().numpy() if torch.is_tensor(value) else value for key, value in output.items()}
            assert all(np.isfinite(value).all() for value in data.values() if isinstance(value, np.ndarray))
            assert data["posed_joints"].shape == (duration * 30, 77, 3)
            rotations = data["global_rot_mats"]
            sample["rotation_orthogonality_error"] = float(np.abs(rotations.swapaxes(-1, -2) @ rotations - np.eye(3)).max())
            sample["minimum_rotation_determinant"] = float(np.linalg.det(rotations).min())
            assert sample["rotation_orthogonality_error"] < 0.001
            assert sample["minimum_rotation_determinant"] > 0.999
            sample["root_displacement_m"] = float(np.linalg.norm(data["root_positions"][-1] - data["root_positions"][0]))
            joints = data["posed_joints"]
            sample["hip_vertical_range_m"] = float(np.ptp(joints[:, skeleton.bone_index["Hips"], 1]))
            feet = joints[:, skeleton.foot_joint_idx]
            contact = data["foot_contacts"] > 0.5
            planted = contact[1:] & contact[:-1]
            speed = np.linalg.norm(np.diff(feet[:, :, [0, 2]], axis=0), axis=-1) * model.fps
            sample["contact_foot_speed_mean_mps"] = float(speed[planted].mean()) if planted.any() else None
            sample["contact_foot_speed_p95_mps"] = float(np.percentile(speed[planted], 95)) if planted.any() else None
            sample["foot_min_y_m"] = float(feet[:, :, 1].min())
            save_kimodo_npz(str(args.output / f"{name}.npz"), data)
            save_motion_bvh(args.output / f"{name}.bvh", output["local_rot_mats"], output["root_positions"],
                            skeleton=skeleton, fps=model.fps, standard_tpose=True)
            sample["frames"] = duration * 30
            sample["joints"] = 77
    del model, skeleton, output
    gc.collect()
    torch.cuda.empty_cache()
    report["allocated_after_unload_gib"] = torch.cuda.memory_allocated() / 2**30

(args.output / f"{args.stage}.json").write_text(json.dumps(report, indent=2) + "\n")
