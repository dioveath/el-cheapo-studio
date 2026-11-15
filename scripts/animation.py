import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
from studio.pipeline.motion import Motion

BASE = os.getenv("STUDIO_URL", "http://localhost:3000")
results = {}


def request(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(15):
        try:
            with urllib.request.urlopen(urllib.request.Request(BASE + "/api" + path, data=data, headers={"Content-Type": "application/json"}), timeout=30) as response:
                return json.load(response)
        except urllib.error.URLError:
            if body is not None or attempt == 14:
                raise
            time.sleep(1)


def wait(job_id, predicate, timeout=300):
    deadline = time.monotonic() + timeout
    previous = None
    while time.monotonic() < deadline:
        job = request("/jobs/" + job_id)
        state = (job["status"], job["phase"], job["attempt"])
        if state != previous:
            print(job_id, state, flush=True)
            previous = state
        assert sum(item["status"] == "running" for item in request("/jobs?limit=200")["jobs"]) <= 1
        if predicate(job):
            return job
        if job["status"] in {"completed", "failed", "cancelled"}:
            raise AssertionError(job)
        time.sleep(0.1)
    raise AssertionError("Job timed out")


def docker(*arguments):
    return subprocess.run(["docker", "compose", *arguments], cwd=ROOT, check=True, capture_output=True, text=True).stdout


library = request("/jobs?limit=200")["jobs"]
assert not any(job["status"] not in {"completed", "failed", "cancelled"} for job in library), "Wait until the studio is idle."
source_id = os.getenv("STUDIO_SOURCE_JOB_ID")
source = next((job for job in library if job["status"] == "completed"
               and job["operation"] == "mesh"
               and (job["id"] == source_id if source_id else job["filename"] == "character.png")), None)
if source is None:
    raise SystemExit("Run scripts/smoke.py first, or set STUDIO_SOURCE_JOB_ID to a completed humanoid mesh job.")
(ROOT / "artifacts").mkdir(exist_ok=True)
original = ROOT / "data/originals" / source["reconstruction_key"] / "original.glb"
original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
settings = {"prompt": "A person walks forward at a relaxed, steady pace.", "duration": 5, "steps": 50, "seed": 42, "root_motion": "in_place"}
shape = request("/jobs", {"asset_ids": [source["asset_id"]], "settings": {"seed": int(time.time()) % 4_000_000_000, "steps": 10, "resolution": 128, "target_vertices": 500, "rig": True}})["jobs"][0]
shape = wait(shape["id"], lambda job: job["status"] == "completed")
assert shape["stats"]["rigging"]["bones"] == 22 and "walk-glb" not in shape["files"]
assert not shape["stats"]["cpu_offload"]
results["asset_with_rig"] = {"id": shape["id"], "stats": shape["stats"]}

motion = settings | {"prompt": "A person makes a gentle bow and raises one hand in greeting.", "seed": int(time.time()) % 4_000_000_000}
job = request(f"/jobs/{shape['id']}/animations", motion)["jobs"][0]
wait(job["id"], lambda item: item["phase"] in {"Encoding motion prompt", "Loading motion model"})
request(f"/jobs/{job['id']}/cancel", {})
cancelled = wait(job["id"], lambda item: item["status"] == "cancelled")
assert cancelled["preview_url"] is None and not cancelled["files"]
processes = docker("exec", "-T", "worker", "ps", "-eo", "args")
assert "/opt/kimodo/bin/python -I /app/studio/pipeline/motion.py" not in processes
retry = request(f"/jobs/{job['id']}/retry", {})["jobs"][0]
retry = wait(retry["id"], lambda item: item["status"] == "completed")
assert retry["parent_id"] == shape["id"] and retry["motion"] == motion
assert not retry["stats"]["motion"]["cpu_offload"]
results["cancel_retry"] = {"cancelled": job["id"], "retried": retry["id"], "stats": retry["stats"]}

motion_cache = Motion(ROOT / "data", motion)
checkpoint = motion_cache.folder / "motion.npz"
checkpoint_time = checkpoint.stat().st_mtime_ns
crash = request(f"/jobs/{retry['id']}/animations", motion | {"root_motion": "travel"})["jobs"][0]
wait(crash["id"], lambda item: item["phase"] == "Retargeting character motion")
command = '''import os,signal,sys,time
from pathlib import Path
for _ in range(100):
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=path.read_bytes().split(b'\\0')
            if args[0].endswith(b'/blender') and any(sys.argv[1].encode() in arg for arg in args):
                os.kill(int(path.parent.name),signal.SIGKILL)
                print('Stopped test export',path.parent.name)
                sys.exit(0)
        except (FileNotFoundError,ProcessLookupError,PermissionError):
            pass
    time.sleep(.02)
raise SystemExit('Export process not found')'''
docker("exec", "-T", "worker", "python", "-c", command, crash["id"])
crash = wait(crash["id"], lambda item: item["status"] == "completed")
assert crash["attempt"] == 2 and crash["stats"]["motion"]["reused"]
assert checkpoint.stat().st_mtime_ns == checkpoint_time
results["export_retry"] = {"id": crash["id"], "attempt": crash["attempt"], "motion_checkpoint_reused": True}

restart = request(f"/jobs/{retry['id']}/animations", motion | {"seed": motion["seed"] + 1, "duration": 10, "steps": 100})["jobs"][0]
wait(restart["id"], lambda item: item["phase"] == "Loading motion model")
docker("kill", "-s", "SIGKILL", "worker")
docker("up", "-d", "--no-deps", "worker")
restart = wait(restart["id"], lambda item: item["status"] == "completed")
assert "animation-glb" in restart["files"]
results["worker_restart"] = {"id": restart["id"], "status": restart["status"]}

shape = request("/jobs", {"asset_ids": [source["asset_id"]], "settings": {"seed": int(time.time()) % 4_000_000_000, "steps": 10, "resolution": 128, "target_vertices": 500}})["jobs"][0]
shape = wait(shape["id"], lambda item: item["status"] == "completed")
assert not shape["stats"]["cpu_offload"]
results["shape_after_motion"] = {"id": shape["id"], "stats": shape["stats"]}
assert hashlib.sha256(original.read_bytes()).hexdigest() == original_hash
results["source_unchanged"] = True
(ROOT / "artifacts/animation-workflow.json").write_text(json.dumps(results, indent=2) + "\n")
print("PASS: asset→rig→Kimodo, GPU serialization, process cancellation/retry, retarget crash recovery, worker restart, checkpoint reuse, Hunyuan reload, unchanged source.", flush=True)
