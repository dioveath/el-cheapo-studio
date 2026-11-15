"""Exercise video generation through the real API, workflow, and GPU worker."""
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from uuid import uuid4

URL = os.getenv("STUDIO_URL", "http://localhost:3000")
ROOT = Path(__file__).resolve().parents[1]


def request(path, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    with urllib.request.urlopen(urllib.request.Request(URL + path, data=data,
        headers={"Content-Type": "application/json"} | (headers or {})), timeout=30) as response:
        return json.load(response)


def wait(job):
    deadline = time.monotonic() + 600
    phase = None
    while time.monotonic() < deadline:
        job = request(f"/api/jobs/{job['id']}")
        if job["phase"] != phase:
            phase = job["phase"]
            print(job["id"], phase, flush=True)
        if job["status"] in {"failed", "cancelled"}:
            raise AssertionError(job["error"] or job["status"])
        if job["status"] == "completed":
            return job
        time.sleep(1)
    raise TimeoutError(job["id"])


def verify(job):
    stats = job["stats"]["video"]
    assert stats["cpu_offload"] is False
    assert (stats["width"], stats["height"], stats["frames"], stats["fps"]) == (320, 192, 81, 16)
    assert all(devices == ["cuda:0"] for check in stats["device_checks"] for devices in check["parameter_devices"].values())
    with urllib.request.urlopen(URL + job["files"]["mp4"]) as response:
        assert response.headers["Content-Type"] == "video/mp4"
        assert response.read(12)[4:8] == b"ftyp"
    with urllib.request.urlopen(urllib.request.Request(URL + job["preview_url"], headers={"Range": "bytes=0-31"})) as response:
        assert response.status == 206
        assert len(response.read()) == 32
    with urllib.request.urlopen(URL + job["thumbnail_url"]) as response:
        assert response.read(8) == b"\x89PNG\r\n\x1a\n"
    return {"id": job["id"], **stats}


def main():
    prompt = "A small red toy car drives steadily from left to right across a plain gray tabletop. Its wheels rotate. Fixed wide side view, soft daylight, one continuous shot."
    text = wait(request("/api/videos", {"settings": {"prompt": prompt, "seed": 42}})["jobs"][0])
    results = [verify(text)]
    boundary = uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="video-chair.png"\r\nContent-Type: image/png\r\n\r\n'.encode()
            + (ROOT / "examples/chair.png").read_bytes() + f"\r\n--{boundary}--\r\n".encode())
    with urllib.request.urlopen(urllib.request.Request(URL + "/api/assets", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})) as response:
        asset = json.load(response)
    image = wait(request("/api/videos", {"asset_id": asset["id"], "settings": {
        "prompt": "A wooden chair rotates slowly to the right. Fixed camera, plain background, one continuous shot.", "seed": 42}})["jobs"][0])
    results.append(verify(image))
    assert {result["mode"] for result in results} == {"text_to_video", "image_to_video"}
    for route in ("variants", "rig", "animations"):
        try:
            request(f"/api/jobs/{text['id']}/{route}", {"prompt": "Walk forward"} if route == "animations" else {})
        except urllib.error.HTTPError as error:
            assert error.code == 409
        else:
            raise AssertionError("Video accepted by mesh operation")
    try:
        request("/api/videos", {"settings": {"prompt": prompt, "width": 1280}})
    except urllib.error.HTTPError as error:
        assert error.code == 422
    else:
        raise AssertionError("Untested resolution accepted")
    destination = ROOT / "artifacts/video-check.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
