# Historical verification notes

These notes record local test runs from November 13–14, 2026, including earlier UI and pipeline versions. Test counts, job IDs, and deployment descriptions refer to those runs. Generated artifacts are excluded from the repository. See [the guide](docs/guide.md#validation) for current validation commands.
Hardware: NVIDIA RTX 5060, 8 GB VRAM. Shape weights remain on CUDA throughout inference.

## Production animation studio

This run tested the Kimodo integration and the earlier full-screen workspace. Model weights stayed on CUDA during inference; CPU geometry processing and Blender exports ran separately.

- Production Docker builds and Next.js TypeScript checks passed.
- **23 Python tests passed**, including process cancellation/reaping, native work draining, motion cache identity, invalid motion settings, geometry, imports, and reconstruction caching.
- Actual `AssetWorkflow` generated and rigged a new character (`7602cb50-ba06-4239-9fbe-0da6229c99b1`).
- Cancellation stopped the supervised Kimodo process before releasing the GPU slot. Explicit retry completed (`01697799-f757-470e-b4ef-25616ada3879`) with the same source and settings.
- Deliberately killing the test's Blender process triggered Temporal attempt 2 and completed (`39bff9d3-99f9-4ca6-87a4-d56a759abdf0`). The saved motion checkpoint's modification time did not change.
- Killing the worker during a ten-second / 100-step motion stage triggered heartbeat recovery and motion attempt 2. The clip completed (`9474b905-e798-4ac6-81de-c10984fc6294`). A subsequent fresh Hunyuan generation also completed (`339dcf33-893f-47c8-9ec5-52894419008e`). Source reconstruction checksums remained unchanged.
- Chromium passed real variant and animation creation, full-window canvas, browser fullscreen, root-following playback, pause stability, scrub/speed controls, skeleton, bind comparison, FBX download, library search, light/dark themes, and 390px mobile layout. No browser errors. Screenshots and the result record are under `artifacts/studio-*`.
- Independent Blender imports of bind and animated GLB/FBX verified 22 bones, 1–4 normalized weights per vertex, matching first-frame geometry, finite coordinates, visible deformation, and expected duration. Five-second clips contain 150 samples at 30 FPS; their first-to-last sample interval is 4.9667 seconds.
- A travelling walk's measured foot-contact speed fell from **0.0871 to 0.0260 scene units/s** after target IK; mean contact target error was 0.00843 scene units. This is one character/motion measurement, not a universal quality guarantee.
- In-place jumps completed on the standard humanoid (`cd839a4c-3ac5-4d6c-9ca0-4ac63e49155c`) and chunky character (`d9d6f02c-4422-41d2-82da-422babbc3f8a`). Independent export sampling measured about **0.443** and **0.149** scene units of vertical hip motion, with horizontal displacement below 1e-4. The chunky character reused the humanoid's generated motion cache.

Evidence: `artifacts/animation-workflow.json`, `artifacts/animation-jumps.json`, `artifacts/studio-browser.json`, and each checked animation folder's `validation.json`/`poses.png`. The full recovery script is `scripts/animation.py`; browser coverage is consolidated in `scripts/browser.cjs`; `scripts/rigging.py` validates production exports.

Known limits: the tested GPU is 8 GB, not 6 GB. NF4 text encoding has not been compared with BF16. Foot-contact correction reduces drift but cannot repair poor skin weights or extreme proportions. The chunky character still needs visual inspection and may need weight editing. Travelling clips retain root translation in exports; the viewer follows their horizontal root motion. Motion endpoints are not guaranteed to form seamless loops.

## Initial asset studio verification

The following records the earlier asset-only implementation.

Passed:

- Production Docker build and TypeScript checks.
- 14 Python tests: topology-aware reduction, geometric/render vertex handling, exports, reconstruction keys, attention projection cache, mesh imports, and parameter limits.
- Real Hunyuan character + chair batch, serialized through Temporal: both reduced to 1,000 geometric vertices with closed surfaces.
- GLB, OBJ, and original downloads; 500-vertex variants reuse the saved reconstruction.
- Faceted GLB import recovers shared geometric vertices and passes through reduction; opaque PNG background removal and generation also completed.
- Unsupported/corrupt uploads, invalid vertex budgets, 12 MB multipart upload through Next, and rejection above 50 MB.
- Failed-job reporting, explicit retry, queued cancellation, and active generation cancellation.
- API/worker restart with a queued variant: saved original reused without inference.
- Hard kill during a generation: heartbeat timeout triggered attempt 2, which completed successfully.
- Chromium: dark/light themes, orbitable WebGL preview, original comparison, wireframe/reset controls, OBJ download, variant creation, example upload/removal, and 390 px mobile layout. No browser errors.

Observed during initial integration:

| Operation | Time | Geometry |
| --- | ---: | ---: |
| Character, first run (includes preprocessing import startup) | 67.84 s | 1,000 vertices / 2,000 triangles |
| Chair, warm worker | 43.68 s | 1,000 vertices |
| Character variant from saved original | 4.47 s | 500 vertices / 1,000 triangles |

CUDA peak allocation was 5.186 GiB; peak reservation 5.613 GiB for both default 30-step, resolution-256 generations. These are PyTorch metrics, not total device usage. No claim of compatibility with a separate 6 GB GPU is made. The final build skips importing the background-removal pipeline for already-transparent images.

Screenshots are in ignored `artifacts/`; machine-readable generation metrics are in `data/verification.json`. Crash/restart results are in `artifacts/recovery.json`. Tests create identifiable example jobs. First-use model downloads and background-removal initialization can add time.


## Workbench redesign — 2026-11-13

- 13 geometry/performance/import tests passed, including thumbnail framing, transparency, scale/translation invariance, and unchanged input geometry. Two tests for the unused in-memory prototype cache were removed with that cache.
- Existing completed jobs received actual mesh thumbnails without running Hunyuan again. Every thumbnail endpoint returned a valid PNG.
- Production builds and TypeScript checks passed.
- Browser coverage verifies one shelf card per source, version/settings restoration, cached variants, mesh thumbnails, wireframe rendering, original comparison, named OBJ export, native dialog focus/Escape behavior, both themes, batch upload, queue cancellation/retry, search, and mobile overflow.
- Thin provider/API/utility wrappers and unused UI components/dependencies were removed. Shared shadcn primitives now live in one controls module. Job mutations return a consistent collection.
- Screenshots: `artifacts/workbench-light.png`, `workbench-dark.png`, `workbench-export.png`, and `workbench-mobile.png`.
- Final browser pass additionally checks thumbnail image bounds and waits for the actual mobile mesh to render. The inspector now sizes to its content.
- Runtime dependency audit: zero advisories. The shadcn CLI is correctly classified as a development dependency.

## Activity resource sharing — 2026-11-13

`scripts/resources.py` ran against the existing Temporal server on an isolated task queue, with the production GPU worker temporarily stopped and restored afterwards. It used the actual Hunyuan weights and a saved character mesh. It did not generate a new shape, load Make-It-Animatable, or measure rigging quality or rigging VRAM.

Eight alternating measurements per configuration, after warmup:

| Measurement | Result |
| --- | ---: |
| One activity: mesh preparation and artifact consumption | 36.03 ms median |
| Two activities: the same work and artifact handoff | 48.27 ms median |
| Additional activity boundary | 12.23 ms |
| Hunyuan live tensor allocation | 4,918.54 MiB |
| Live tensor allocation after releasing the model | 0 MiB |
| Reloading Hunyuan from the local cache | 17.00 s |

Both activities retained the same process, model object, CUDA parameter pointer and tensor allocation. Four concurrent workflows produced zero overlapping activity executions with one worker and `max_concurrent_activities=1`. Their stages interleaved across jobs, so that setting does not guarantee whole-job exclusivity or prevent repeated model switches. An intentional downstream failure retried that activity once; mesh preparation executed only once and the saved mesh remained usable.

Two exploratory runs first loaded Hunyuan inside the activity executor, then unloaded/reloaded it on the main thread. Their measurements completed, but the processes aborted during shutdown with a native allocator error. Explicitly releasing the final model did not resolve it. Loading Hunyuan on the main thread before starting the worker, matching production startup, completed with exit code 0. The exact native cause is unconfirmed; the successful probe does not establish that arbitrary cross-thread model loading is safe. Independent checks with 30 serialized mesh reductions, both with and without Hunyuan dependency imports, also exited cleanly.

Recommendation: keep mesh preparation and optional rigging as separate, substantial activities in one GPU worker. Persist artifact paths between stages, keep model lifecycle ownership in the worker, and serialize GPU use. Avoid retaining both models until MIA memory measurements justify it, and account for model-switch costs when scheduling batches. These measurements alone do not establish MIA dependency compatibility or animation quality.

To reproduce, place a saved character GLB at `data/resource-probe/character.glb`, wait for active jobs to finish, stop the production worker, and run the worker image with `scripts/resources.py` mounted at `/app/resources.py`, invoking `python /app/resources.py`. Always restart the production worker afterwards. Results are written to ignored `data/resource-probe/results.json`; the probe uses a unique Temporal queue on every run. GPU memory figures above are PyTorch live tensor allocations, not total device memory usage.

## Rigging and first animation — 2026-11-13

Implemented and deployed at the same LAN address. The workflow now runs `prepare_mesh`, optional `rig_character`, and `finish_job`, with durable artifact checkpoints and one activity at a time. Existing exports migrated to the artifact map; a fresh database and a repeated initialization also passed.

Final validation:

- 15 Python tests pass, including cancellation draining and MIA import after mesh-processing libraries without loading `bpy`.
- Production worker/web images build; TypeScript checks pass. The running worker uses the checked-in source.
- Real 30-step, resolution-128 Hunyuan generation → MIA rigging → Hunyuan generation passes without weight offload. Shape peak allocation: 5.186–5.194 GiB; reservation: 5.586–5.598 GiB. Model switches leave approximately 0.008 GiB of PyTorch allocations, rather than retaining the previous model's weights.
- A forced SIGKILL of the test job's Blender export caused exactly one rigging activity retry. The prediction checkpoint's timestamp stayed unchanged. Temporal history contains one mesh stage, a rigging stage completing on attempt 2, and final completion. The original and reduced source files stayed unchanged.
- Active export cancellation terminated the child process. Explicit retry retained the rig operation, source parent, and reconstruction. The API did not expose cancelled outputs.
- Graceful worker shutdown after both models had been used exited with code 0; no automatic container restarts.
- Chromium passes both the original workbench regression and the new rigging flow: create a rig, play/pause, scrub, skeleton, static/original comparison, GLB/FBX/OBJ downloads, light/dark themes, keyboard dialogs, variants, batch queue/cancel/retry, search, and 390 px layout. No browser errors.
- Independent Blender reimports of all four exports pass for both characters: 22 bones, normalized weights, at most four influences, an armature modifier, no animation in the rest export, and a changing walk pose in the animated export. Rest GLB/FBX geometry matches the saved input within 0.000003 units. Faceted GLB has split rendering vertices; FBX retains shared geometry.

| Example | Geometric vertices | MIA inference | Peak allocated / reserved | Result |
| --- | ---: | ---: | ---: | --- |
| Chunky character | 1,000 | 0.55 s | 2.75 / 2.79 GiB | Valid rig and walk; visible lower-body/weight deformation |
| Standard humanoid | 506 | 0.69 s | 2.75 / 2.80 GiB | Recognizable walk with arms lowered from the source pose |

The humanoid's first rig job, including model loading and export, took 6.76 seconds. The 1.38-second retry measurement covers only the successful export attempt, not the original inference or retry delay. Both examples have a 30-frame, approximately 0.967-second looping clip. Measurements are from the RTX 5060 with 8 GB; they do not establish operation on a separate 6 GB GPU.

Quality limits: no foot-contact solver, independent tail/finger bones, facial animation, or weight editing. The short-legged character is animatable but its walk is not polished. Structural validation does not imply good deformation on every mesh. The source bind geometry remains intact; the walking export intentionally changes pose. Retargeting follows joint-chain directions because the reference's imported bone axes do not consistently follow its posed child joints. Other animation clips still require retargeting.

Resolved during integration:

- Python `bpy` crashed after export and conflicted with PyMeshLab's Embree library when imported after generation. Blender now runs only as a standalone process. The pinned MIA source patch reads skeleton metadata generated at image build time and defers Blender-only dataset imports; the Python Blender package and its wheel-metadata workaround were removed.
- The async Hunyuan loader's retained frame held a full model after switching. Its returned model and final module references are explicitly cleared from that frame; the real switching test confirms release.
- Upstream weight assignment discarded small nonzero weights. Export now writes every retained influence directly, preserving normalization on both test characters.
- The viewer now computes precise skinned bounds after updating bone transforms, avoiding undersized, displaced previews.

Evidence: `artifacts/animation-workflow.json`, `rigging-history.json`, `rigging-shutdown.json`, `rigging-tests.log`, and `animation-browser.log`. Screenshots are `artifacts/rigging-{light,dark,skeleton,export,mobile}.png`. Final sample jobs are `db98fb2c-7c33-4c56-823d-4801972efae7` (chunky character) and `53e6e877-ddae-4736-8427-7c84a70a465b` (humanoid); their `rigging/` folders contain `validation.json` and `poses.png`. Test failures and cancelled jobs remain in history.

## Kimodo benchmark — 2026-11-13

Kimodo-SOMA-RP-v1.2 runs on this RTX 5060 8 GB without model-weight CPU offloading. The benchmark uses NVIDIA source `e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954`, motion weights `0e3c89024d896064b6851201373f7834784a386b`, and the community `matbee/kimodo-llm2vec-nf4` encoder at `268b2f7bbbb8361bce76cc3f34d61b5467643a59`. This is the quantized route, not NVIDIA's stock BF16 text encoder. Both models' parameters were asserted to reside entirely on `cuda:0`. The inference containers ran with networking disabled after downloading the pinned weights.

Text encoding and motion generation ran in separate processes. The encoder produced three cached 4096-value prompt embeddings; its weights were released before loading the motion model. CPU motion correction and Blender export remain part of the pipeline. No full model stays on CPU during inference.

Median motion generation time over three seeds, after a warmup, including NVIDIA motion correction but excluding text encoding, loading, and exports:

| Clip duration | 50 steps | 100 steps |
| --- | ---: | ---: |
| 3 seconds | 2.09 s | 4.20 s |
| 5 seconds | 2.86 s | 5.71 s |
| 10 seconds | 6.54 s | 13.09 s |

Memory and startup:

- Maximum sampled total device usage: 5,076 MiB / 4.96 GiB, during text encoding. CUDA free-memory sampling ran every 20 ms; independent `nvidia-smi` sampling ran every 100 ms and agreed on the peak.
- Encoder peak PyTorch allocation: 4.65 GiB; peak reservation: 4.81 GiB. Motion peak total device usage: 1.36 GiB.
- Loading cached encoder weights: 3.31 s; motion weights: 0.64 s. Warm text encoding: 0.138 s median. These exclude Python imports/container startup; the encoder process, including nine encodings, took 8.98 s.
- Both stages released weights, leaving 0.008 GiB allocated. Peak process host RSS: 5.30 GiB for encoding and 3.01 GiB for motion.
- Only the actual 8 GB card was tested. These measurements suggest room on a 6 GB card, but do not establish compatibility with different hardware. Co-resident encoder/motion models were not benchmarked.

Validation and quality:

- All 24 measured clips and the warmup completed: 18 walk samples across the duration/step matrix, plus three jumps and three waves at five seconds / 100 steps. All NPZ/BVH exports contain finite data and 77-joint motion. Rotation matrices pass orthogonality/determinant checks. Repeated prompt encodings were identical.
- Six retargets reuse the existing humanoid and chunky rigs. All 12 GLB/FBX exports independently reimport with 22 bones, normalized weights, at most four influences, and correct duration. First-pose round-trip error stays below 0.00001 scene units. Input rig hashes remain unchanged.
- Humanoid walking, jumping and waving are recognizable. The retarget preserves vertical hip motion while removing horizontal travel. A coordinate-space issue that initially suppressed jump height was corrected; the humanoid jump now has 0.444 scene units of vertical hip travel.
- The chunky character retains substantial limb/body deformation from its rig and the adaptation to humanoid motion. The benchmark does not establish production quality for that character. It does not add target-character foot IK or improve skin weights.
- On the source SOMA skeleton, five-second / 100-step walks retain 2.3–3.3 cm/s mean horizontal speed during labeled foot contact after correction. Lowest foot joints range down to approximately 1.5 cm below the ground for these walks. Source motion correction does not guarantee contact after retargeting to different proportions.
- No full-precision reference comparison or formal text-motion retrieval benchmark was performed. The sample results do not establish that NF4 preserves all motion quality.

The benchmark uses an isolated image with PyTorch 2.7.1 / CUDA 12.8, Transformers 5.1.0, bitsandbytes 0.49.2, PEFT 0.18.1 and Accelerate 1.12.0. Native motion correction builds against pybind11 2.11.1; the image uses system CMake to avoid the base image's Python shim. No community runtime monkey patches are needed. The encoder restores the original Llama model identity for upstream chat-token preparation. The SOMA 77-joint output has six foot-contact channels, so measurements follow the skeleton metadata rather than assuming the four-channel internal layout.

Evidence is in ignored `artifacts/kimodo/`: `encode.json`, `motion.json`, `gpu.csv`, `processes.json`, `retarget.json`, raw NPZ/BVH clips, and `humanoid/` / `chunky/` GLB, FBX, GIF and PNG previews. Production code, Temporal workflows and UI are unchanged. The production worker was restored and the API and worker health checks passed.

Recommendation: proceed with a separate motion activity and persisted motion artifacts, keeping GPU scheduling shared with shape generation and rigging. Start with the measured sequential loading strategy. Fifty steps is a useful preview setting; retain 100 steps as an option. Improve chunky-character rigging and add target foot-contact handling before treating all generated characters as animation-ready.


## LTX-Video integration — 2026-11-14

Added a Video workspace with prompt-only and optional-image generation, native MP4 playback/download, collection entries, and the existing Temporal queue/cancel/retry flow. The preset is LTX-Video 2B 0.9.8 distilled FP8 with T5 Q3_K_S, 320 × 192, 81 frames at 16 FPS, eight Euler steps, CFG 1. Model files and ComfyUI/GGUF revisions are pinned; the video runtime is isolated from Hunyuan, MIA, and Kimodo.

Real API → Temporal → GPU tests on the RTX 5060 8 GB:

| Input | Job | Duration | Peak total board VRAM | Generation including model load |
| --- | --- | --- | --- | --- |
| Text | `ac0f3d09-0dec-44ef-8884-08798fa5f52a` | 5.0625 s | 7.106 GiB | 15.06 s |
| Image + text | `932c398e-f080-45e3-9321-bced8d3faca2` | 5.0625 s | 7.108 GiB | 12.75 s |

Both outputs independently decoded to 81 valid 320 × 192 frames. Transformer, text encoder, and VAE parameter audits reported CUDA at load, prompt encoding, sampling, and decoding; image encoding was also audited. CPU offloading and automatic model eviction are disabled. Peak CUDA tensor allocation was 6.373 GiB in both runs; board figures include desktop usage. These are small-preset measurements, not a guarantee for every prompt or when other GPU applications are active. Motion adherence remains imperfect.

Validation: all 28 backend tests passed, Next.js production build and TypeScript passed, and the real browser generated a clip after cancelling/retrying a video job. Playback, pause, seeking, MP4 download, image upload/removal, new-video reset, desktop light/dark, and 390 px mobile layout passed with no browser errors. HTTP range requests returned 206; unsupported video resolutions returned 422; mesh, rig, and animation endpoints rejected video jobs with 409.

Evidence: `artifacts/video-check.json`, `artifacts/video-browser.json`, and `artifacts/video-{light,dark,mobile}.png`. Reproduce with `scripts/video-check.py` and `scripts/video-browser.cjs` (Playwright 1.65.0). Existing library entries were preserved.


## Workspace UI refresh — 2026-11-14

Reworked Create, Animate, and Video into a consistent workspace with a framed preview, a separate settings column, a direct download action, and recent-work thumbnails. Mobile uses a stacked layout without controls covering the output. Added a searchable modal library with type filters, Escape handling, focus restoration, prompt ideas, a starting-image drop zone, seed randomization, prompt preservation across workspace changes, and Ctrl/Cmd+Enter submission. Animation thumbnails display their motion prompts. Light and dark themes use the same layout.

The production Next.js build and TypeScript checks passed. `scripts/ui-review.cjs` passed with no page errors: prompt switching/preservation, image upload/removal, seed randomization, MP4 download, library filtering and keyboard return focus, 390 px and 320 px layouts with no horizontal overflow, 3D preview bounds, animation playback/pause/scrub, and mesh budget selection. Screenshots were inspected for empty and completed states, video, animation, library, both themes, and mobile. Evidence: `artifacts/ui-review.json` and `artifacts/ui-*.png`.

The redesigned UI also passed a real video generation/cancel/retry/download run (`artifacts/video-browser.json`, completed job `e3153807-2813-4dee-9a12-2f818867d749`, 81 frames at 16 FPS). A 1366 × 768 check was added to keep the generation action visible and ensure the player stays inside its preview with its original aspect ratio. Generation actions stay visible while the desktop settings panel scrolls.
