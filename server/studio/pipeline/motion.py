import argparse
import hashlib
import json
from pathlib import Path
import time

MODELS = {
    "encoder": ("matbee/kimodo-llm2vec-nf4", "268b2f7bbbb8361bce76cc3f34d61b5467643a59"),
    "motion": ("nvidia/Kimodo-SOMA-RP-v1.2", "0e3c89024d896064b6851201373f7834784a386b"),
}
SOURCE_REVISION = "e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954"


class Motion:
    def __init__(self, root, settings):
        self.settings = settings
        encoded = json.dumps([MODELS["encoder"], SOURCE_REVISION, settings["prompt"]], separators=(",", ":"))
        self.embedding = root / "embeddings" / hashlib.sha256(encoded.encode()).hexdigest()
        generated = json.dumps([encoded, MODELS["motion"], settings["duration"], settings["steps"], settings["seed"]])
        self.folder = root / "motions" / hashlib.sha256(generated.encode()).hexdigest()

    def run(self, stage):
        import numpy as np
        import torch
        from huggingface_hub import snapshot_download
        from kimodo.sanitize import sanitize_texts

        started = time.monotonic()
        torch.set_num_threads(8)
        torch.cuda.reset_peak_memory_stats()
        repo, revision = MODELS["encoder" if stage == "encode" else "motion"]
        try:
            path = Path(snapshot_download(repo, revision=revision, local_files_only=True))
        except FileNotFoundError:
            path = Path(snapshot_download(repo, revision=revision, max_workers=2))
        text = sanitize_texts([self.settings["prompt"]])[0]
        if stage == "encode":
            from kimodo.model.llm2vec.llm2vec import LLM2Vec

            encoder = LLM2Vec.from_pretrained(
                base_model_name_or_path=str(path), peft_model_name_or_path=str(path / "supervised_adapter"),
                torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}, low_cpu_mem_usage=True,
            ).eval().requires_grad_(False)
            encoder.model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
            if {str(parameter.device) for parameter in encoder.parameters()} != {"cuda:0"}:
                raise RuntimeError("The text encoder must fit entirely on the GPU.")
            with torch.inference_mode():
                sentence = encoder._convert_to_str("", text)
                features = encoder.tokenize([encoder.prepare_for_tokenization(sentence)])
                vector = encoder({key: value.to("cuda") for key, value in features.items()}).float().cpu().numpy()[0]
            if vector.shape != (4096,) or not np.isfinite(vector).all():
                raise ValueError("Kimodo returned an invalid text embedding.")
            self.embedding.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.embedding.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.save(stream, vector)
            temporary.replace(self.embedding)
        else:
            from hydra.utils import instantiate
            from omegaconf import OmegaConf
            from kimodo.exports.bvh import save_motion_bvh
            from kimodo.exports.motion_io import save_kimodo_npz
            from kimodo.tools import seed_everything

            config = OmegaConf.load(path / "config.yaml")
            config.checkpoint_dir = str(path)
            config = OmegaConf.to_container(config, resolve=True)
            config.pop("checkpoint_dir")
            model = instantiate(config, text_encoder=None, device="cuda").eval().requires_grad_(False)
            embedding = torch.from_numpy(np.load(self.embedding)).cuda()[None, None]
            model.text_encoder = lambda prompts: (embedding.expand(len(prompts), -1, -1), [1] * len(prompts))
            seed_everything(self.settings["seed"])

            def progress(steps):
                for index, step in enumerate(steps):
                    print("STUDIO " + json.dumps({"phase": f"Generating motion · step {index + 1}/{self.settings['steps']}",
                                                  "progress": 35 + 45 * (index + 1) / self.settings["steps"]}), flush=True)
                    yield step

            with torch.inference_mode():
                output = model([text], [self.settings["duration"] * model.fps],
                               num_denoising_steps=self.settings["steps"], post_processing=True,
                               return_numpy=False, progress_bar=progress)
                output = {key: value[0] for key, value in output.items()}
            data = {key: value.detach().cpu().numpy() if torch.is_tensor(value) else value for key, value in output.items()}
            if not all(np.isfinite(value).all() for value in data.values() if isinstance(value, np.ndarray)):
                raise ValueError("Kimodo returned invalid motion coordinates.")
            self.folder.mkdir(parents=True, exist_ok=True)
            save_kimodo_npz(str(self.folder / "motion.npz"), data)
            save_motion_bvh(self.folder / "motion.bvh", output["local_rot_mats"], output["root_positions"],
                            skeleton=model.output_skeleton, fps=model.fps, standard_tpose=True)
            contacts = data["foot_contacts"]
            names = [model.output_skeleton.bone_order_names[index] for index in model.output_skeleton.foot_joint_idx]
            (self.folder / "contacts.json").write_text(json.dumps({"names": names, "contacts": contacts.tolist()}))
            stats = {"model": MODELS["motion"][0], "model_revision": MODELS["motion"][1],
                     "encoder_revision": MODELS["encoder"][1], "source_revision": SOURCE_REVISION,
                     "cpu_offload": False, "frames": self.settings["duration"] * model.fps, "fps": model.fps,
                     "generation_seconds": round(time.monotonic() - started, 2),
                     "peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)}
            temporary = self.folder / "motion.tmp"
            temporary.write_text(json.dumps(stats))
            temporary.replace(self.folder / "motion.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("encode", "motion"))
    parser.add_argument("root", type=Path)
    parser.add_argument("settings", type=Path)
    args = parser.parse_args()
    Motion(args.root, json.loads(args.settings.read_text())).run(args.stage)
