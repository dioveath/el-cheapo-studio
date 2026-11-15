import asyncio
from contextlib import asynccontextmanager, suppress
import hashlib
import logging
import os
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from psycopg.types.json import Jsonb
from temporalio.client import Client
from temporalio.exceptions import WorkflowAlreadyStartedError
from temporalio.service import RPCError, RPCStatusCode

from .db import DATA_ROOT, connection, get_job, initialize
from .schemas import CreateJobs, CreateVideo, GenerationSettings, MotionSettings, VariantSettings

logger = logging.getLogger(__name__)
TASK_QUEUE = "low-poly-gpu"
TERMINAL = {"completed", "failed", "cancelled"}
MAX_UPLOAD = 50 * 1024 * 1024


def pending_jobs():
    with connection() as conn:
        return conn.execute("SELECT id,operation FROM jobs WHERE NOT dispatched AND status='queued' AND NOT cancel_requested ORDER BY created_at LIMIT 32").fetchall()


async def dispatch(client):
    while True:
        try:
            for row in await asyncio.to_thread(pending_jobs):
                job_id = str(row["id"])
                try:
                    name = {"animation": "AnimationWorkflow", "video": "VideoWorkflow"}.get(row["operation"], "AssetWorkflow")
                    await client.start_workflow(name, job_id, id=f"job-{job_id}", task_queue=TASK_QUEUE)
                except WorkflowAlreadyStartedError:
                    pass
                with connection() as conn:
                    conn.execute("UPDATE jobs SET dispatched=true WHERE id=%s", (job_id,))
                    cancelled = conn.execute("SELECT cancel_requested FROM jobs WHERE id=%s", (job_id,)).fetchone()["cancel_requested"]
                if cancelled:
                    await client.get_workflow_handle(f"job-{job_id}").cancel()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Workflow dispatch will retry")
        await asyncio.sleep(1)


@asynccontextmanager
async def lifespan(app):
    initialize()
    app.state.temporal = await Client.connect(os.getenv("TEMPORAL_ADDRESS", "temporal:7233"))
    task = asyncio.create_task(dispatch(app.state.temporal))
    yield
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


app = FastAPI(
    title="Low Poly Studio",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)


def serialize_job(row):
    if not row:
        raise HTTPException(404, "Job not found")
    result = {key: value for key, value in row.items() if key != "artifacts"}
    job_id = str(row["id"])
    files = {name: f"/api/jobs/{job_id}/files/{name}" for name, path in row["artifacts"].items()
             if name not in {"geometry", "thumbnail"} and Path(path).suffix in {".glb", ".obj", ".fbx", ".bvh", ".mp4"}} if row["status"] == "completed" else {}
    result["files"] = files
    result["preview_url"] = files.get("mp4", files.get("animation-glb", files.get("rig-glb", files.get("glb"))))
    result["original_url"] = f"/api/jobs/{job_id}/files/original" if row.get("reconstruction_key") else None
    result["thumbnail_url"] = f"/api/jobs/{job_id}/files/thumbnail" if row["status"] == "completed" else None
    return result


@app.get("/api/health")
def health():
    with connection() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok", "models": ["Hunyuan3D-2", "Make-It-Animatable", "Kimodo-SOMA-RP-v1.2", "LTX-Video-2B-distilled-FP8-T5-Q3_K_S"], "gpu_concurrency": 1}


@app.post("/api/assets", status_code=201)
async def upload_asset(file: UploadFile = File(...)):
    filename = Path((file.filename or "upload").replace("\\", "/")).name[:180]
    extension = Path(filename).suffix.lower()
    if extension not in {".png", ".jpg", ".jpeg", ".webp", ".glb", ".obj"}:
        raise HTTPException(415, "Use PNG, JPG, WebP, GLB, or OBJ.")
    asset_id = uuid4()
    folder = DATA_ROOT / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{asset_id}{extension}"
    size = 0
    digest = hashlib.sha256()
    try:
        with path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, "Each file must be smaller than 50 MB.")
                output.write(chunk)
                digest.update(chunk)
        if not size:
            raise HTTPException(400, "The uploaded file is empty.")
        kind = "mesh" if extension in {".glb", ".obj"} else "image"
        thumbnail = None
        if kind == "image":
            try:
                with Image.open(path) as image:
                    if image.width * image.height > 25_000_000:
                        raise HTTPException(413, "Images must be at most 25 megapixels.")
                    image = image.convert("RGBA")
                    image.thumbnail((640, 640))
                    thumb = folder / f"{asset_id}.preview.png"
                    image.save(thumb)
                    thumbnail = str(thumb.relative_to(DATA_ROOT))
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                raise HTTPException(400, "This image could not be read.") from exc
        with connection() as conn:
            conn.execute("INSERT INTO assets (id,filename,kind,path,thumbnail,sha256,size_bytes) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                         (asset_id, filename, kind, str(path.relative_to(DATA_ROOT)), thumbnail, digest.hexdigest(), size))
        return {"id": asset_id, "filename": filename, "kind": kind, "size_bytes": size,
                "thumbnail_url": f"/api/assets/{asset_id}/thumbnail" if thumbnail else None}
    except Exception:
        path.unlink(missing_ok=True)
        (folder / f"{asset_id}.preview.png").unlink(missing_ok=True)
        raise
    finally:
        await file.close()


@app.get("/api/assets/{asset_id}/thumbnail")
def asset_thumbnail(asset_id: UUID):
    with connection() as conn:
        asset = conn.execute("SELECT thumbnail FROM assets WHERE id=%s", (asset_id,)).fetchone()
    if not asset or not asset["thumbnail"]:
        raise HTTPException(404, "Thumbnail not found")
    return FileResponse(DATA_ROOT / asset["thumbnail"], media_type="image/png")


@app.get("/api/examples/{name}")
def example(name: str):
    if name not in {"chair", "character"}:
        raise HTTPException(404)
    return FileResponse(Path("/app/examples") / f"{name}.png", media_type="image/png")


@app.post("/api/jobs", status_code=201)
def create_jobs(body: CreateJobs):
    batch = uuid4()
    ids = []
    with connection() as conn:
        for asset_id in body.asset_ids:
            if not conn.execute("SELECT 1 FROM assets WHERE id=%s", (asset_id,)).fetchone():
                raise HTTPException(404, "An uploaded asset was not found.")
            job_id = uuid4()
            conn.execute("INSERT INTO jobs (id,asset_id,batch_id,settings) VALUES (%s,%s,%s,%s)",
                         (job_id, asset_id, batch, Jsonb(body.settings.model_dump())))
            ids.append(job_id)
    return {"batch_id": batch, "jobs": [serialize_job(get_job(job_id)) for job_id in ids]}


@app.get("/api/jobs")
def list_jobs(limit: int = 100):
    with connection() as conn:
        rows = conn.execute("SELECT j.*, COALESCE(a.filename, left(j.settings->>'prompt', 80)) AS filename, COALESCE(a.kind, 'text') AS kind FROM jobs j LEFT JOIN assets a ON a.id=j.asset_id ORDER BY j.created_at DESC LIMIT %s", (min(max(limit, 1), 200),)).fetchall()
        counts = conn.execute("SELECT status, count(*) AS count FROM jobs GROUP BY status").fetchall()
    return {"jobs": [serialize_job(row) for row in rows], "counts": {row["status"]: row["count"] for row in counts}}


@app.post("/api/videos", status_code=201)
def create_video(body: CreateVideo):
    job_id = uuid4()
    with connection() as conn:
        if body.asset_id:
            asset = conn.execute("SELECT kind FROM assets WHERE id=%s", (body.asset_id,)).fetchone()
            if not asset:
                raise HTTPException(404, "The uploaded image was not found.")
            if asset["kind"] != "image":
                raise HTTPException(422, "Video input must be an image.")
        conn.execute("INSERT INTO jobs (id,asset_id,batch_id,settings,operation) VALUES (%s,%s,%s,%s,'video')",
                     (job_id, body.asset_id, uuid4(), Jsonb(body.settings.model_dump())))
    return {"jobs": [serialize_job(get_job(job_id))]}


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: UUID):
    return serialize_job(get_job(job_id))


@app.post("/api/jobs/{job_id}/variants", status_code=201)
def create_variant(job_id: UUID, body: VariantSettings):
    source = get_job(job_id)
    if not source or source["status"] != "completed" or source["operation"] == "video":
        raise HTTPException(409, "Choose a completed job to create a variant.")
    settings = GenerationSettings.model_validate(source["settings"] | body.model_dump())
    new_id = uuid4()
    with connection() as conn:
        conn.execute("INSERT INTO jobs (id,asset_id,parent_id,batch_id,settings) VALUES (%s,%s,%s,%s,%s)",
                     (new_id, source["asset_id"], job_id, uuid4(), Jsonb(settings.model_dump())))
    return {"jobs": [serialize_job(get_job(new_id))]}


@app.post("/api/jobs/{job_id}/rig", status_code=201)
def rig_character(job_id: UUID):
    source = get_job(job_id)
    if not source or source["status"] != "completed" or source["operation"] == "video":
        raise HTTPException(409, "Choose a completed character to rig.")
    artifacts = {key: source["artifacts"][key] for key in ("geometry", "glb", "obj", "thumbnail")}
    if not all((DATA_ROOT / path).is_file() for path in artifacts.values()):
        raise HTTPException(409, "The saved mesh is missing.")
    new_id = uuid4()
    with connection() as conn:
        conn.execute("INSERT INTO jobs (id,asset_id,parent_id,batch_id,settings,operation,reconstruction_key,artifacts,stats) VALUES (%s,%s,%s,%s,%s,'rig',%s,%s,%s)",
                     (new_id, source["asset_id"], job_id, uuid4(), Jsonb(source["settings"]), source["reconstruction_key"],
                      Jsonb(artifacts), Jsonb({key: value for key, value in source["stats"].items() if key != "rigging"})))
    return {"jobs": [serialize_job(get_job(new_id))]}


@app.post("/api/jobs/{job_id}/animations", status_code=201)
def create_animation(job_id: UUID, body: MotionSettings):
    source = get_job(job_id)
    if not source or source["status"] != "completed" or source["operation"] == "video":
        raise HTTPException(409, "Choose a completed character to animate.")
    artifacts = {key: value for key, value in source["artifacts"].items()
                 if key in {"geometry", "glb", "obj", "thumbnail", "rig-glb", "rig-fbx"}}
    if not artifacts.get("geometry") or not all((DATA_ROOT / path).is_file() for path in artifacts.values()):
        raise HTTPException(409, "The saved character is incomplete. Choose another version.")
    new_id = uuid4()
    with connection() as conn:
        conn.execute("INSERT INTO jobs (id,asset_id,parent_id,batch_id,settings,operation,motion,reconstruction_key,artifacts,stats) VALUES (%s,%s,%s,%s,%s,'animation',%s,%s,%s,%s)",
                     (new_id, source["asset_id"], job_id, uuid4(), Jsonb(source["settings"]), Jsonb(body.model_dump()),
                      source["reconstruction_key"], Jsonb(artifacts),
                      Jsonb({key: value for key, value in source["stats"].items() if key not in {"motion", "animation"}})))
    return {"jobs": [serialize_job(get_job(new_id))]}


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: UUID):
    with connection() as conn:
        row = conn.execute("SELECT status,dispatched FROM jobs WHERE id=%s FOR UPDATE", (job_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Job not found")
        if row["status"] in TERMINAL:
            return {"jobs": [serialize_job(get_job(job_id))]}
        status = "cancelling" if row["dispatched"] else "cancelled"
        conn.execute("UPDATE jobs SET cancel_requested=true,status=%s,phase=%s,updated_at=now() WHERE id=%s",
                     (status, "Cancellation requested" if row["dispatched"] else "Cancelled", job_id))
    if row["dispatched"]:
        try:
            await app.state.temporal.get_workflow_handle(f"job-{job_id}").cancel()
        except RPCError as exc:
            if exc.status != RPCStatusCode.NOT_FOUND:
                raise
    return {"jobs": [serialize_job(get_job(job_id))]}


@app.post("/api/jobs/{job_id}/retry", status_code=201)
def retry_job(job_id: UUID):
    source = get_job(job_id)
    if not source or source["status"] not in {"failed", "cancelled"}:
        raise HTTPException(409, "Only failed or cancelled jobs can be retried.")
    new_id = uuid4()
    with connection() as conn:
        conn.execute("INSERT INTO jobs (id,asset_id,parent_id,batch_id,settings,operation,motion,reconstruction_key,artifacts,stats) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                     (new_id, source["asset_id"], source["parent_id"], uuid4(), Jsonb(source["settings"]),
                      source["operation"], Jsonb(source["motion"]) if source["motion"] else None,
                      source["reconstruction_key"], Jsonb(source["artifacts"]), Jsonb(source["stats"])))
    return {"jobs": [serialize_job(get_job(new_id))]}


@app.get("/api/jobs/{job_id}/files/{kind}")
def download(job_id: UUID, kind: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if kind == "original" and job["reconstruction_key"]:
        with connection() as conn:
            original = conn.execute("SELECT original_path FROM reconstructions WHERE key=%s", (job["reconstruction_key"],)).fetchone()
        path = original["original_path"] if original else None
    elif job["status"] == "completed":
        path = job["artifacts"].get(kind) if kind != "geometry" else None
        if path and Path(path).suffix not in {".glb", ".obj", ".fbx", ".bvh", ".png", ".mp4"}:
            path = None
    else:
        path = None
    if not path or not (DATA_ROOT / path).is_file():
        raise HTTPException(404, "This file is not ready.")
    path = DATA_ROOT / path
    media = {".glb": "model/gltf-binary", ".obj": "text/plain", ".png": "image/png", ".mp4": "video/mp4", ".fbx": "application/octet-stream", ".bvh": "application/octet-stream"}
    name = f"{Path(job['filename']).stem}-{kind}{path.suffix}"
    return FileResponse(path, media_type=media[path.suffix], filename=name if kind != "thumbnail" else None)
