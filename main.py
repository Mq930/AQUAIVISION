"""
main.py  –  AQUAVISION FastAPI server
Serves pages, handles upload/process/stream/download.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, shutil, asyncio, threading, logging, json, queue, base64
from pathlib import Path
from typing import Optional, Dict

from inference_module import process_video, process_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────
class ProcessRequest(BaseModel):
    filename:     str
    yolo_enabled: bool = True
    hud_enabled:  bool = True

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="AQUAVISION",
    description="Underwater AI Enhancement & Detection System",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).parent
UPLOAD_DIR   = PROJECT_ROOT / "uploads"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Per-job state  { job_id: { status, frame_queue, progress, result, error, … } }
processing_state: Dict[str, dict] = {}
processing_lock = threading.Lock()

# ─────────────────────────────────────────────
# Page routes
# ─────────────────────────────────────────────
def _serve(name: str):
    p = PROJECT_ROOT / name
    if not p.exists():
        raise HTTPException(404, f"{name} not found")
    return p.read_text(encoding="utf-8")

@app.get("/", response_class=HTMLResponse)
async def home():
    return _serve("index.html")

@app.get("/about", response_class=HTMLResponse)
async def about():
    return _serve("about.html")

@app.get("/upload", response_class=HTMLResponse)
async def upload_page():
    return _serve("upload.html")

@app.get("/results", response_class=HTMLResponse)
async def results_page():
    return _serve("results.html")

# ─────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────
ALLOWED_VIDEO = {".mp4",".avi",".mov",".mkv",".flv",".wmv",".webm"}
ALLOWED_IMAGE = {".jpg",".jpeg",".png",".bmp",".tiff",".webp"}
MAX_SIZE_BYTES = 200 * 1024 * 1024   # 200 MB

@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO | ALLOWED_IMAGE:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    # Read & size-check
    data = await file.read()
    if len(data) > MAX_SIZE_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_SIZE_BYTES//(1024*1024)} MB limit")

    dest = UPLOAD_DIR / file.filename
    dest.write_bytes(data)
    logger.info(f"Uploaded: {file.filename}  ({len(data)//(1024*1024)} MB)")

    return {
        "status":   "success",
        "filename": file.filename,
        "size_mb":  round(len(data) / (1024*1024), 2),
        "type":     "video" if ext in ALLOWED_VIDEO else "image",
    }

# ─────────────────────────────────────────────
# Video processing  (async, with SSE streaming)
# ─────────────────────────────────────────────
@app.post("/api/process")
async def api_process(request: ProcessRequest, background_tasks: BackgroundTasks):
    filename = request.filename
    input_path = UPLOAD_DIR / filename
    if not input_path.exists():
        raise HTTPException(404, f"File not found: {filename}")

    output_filename = f"processed_{Path(filename).stem}.mp4"
    output_path     = OUTPUT_DIR / output_filename
    job_id          = filename

    # Read video metadata so the client can sync the original video
    import cv2 as _cv2
    _cap = _cv2.VideoCapture(str(input_path))
    source_fps    = _cap.get(_cv2.CAP_PROP_FPS) or 30.0
    source_frames = int(_cap.get(_cv2.CAP_PROP_FRAME_COUNT))
    _cap.release()

    # Frame queue — pipeline produces ~53 FPS, SSE consumer drains at 30 FPS.
    # Buffer ~6 s worth of frames (53*6=318) so the pipeline is never blocked.
    fq = queue.Queue(maxsize=320)

    with processing_lock:
        processing_state[job_id] = {
            "status":      "processing",
            "input_file":  filename,
            "output_file": output_filename,
            "progress":    {"current_frame": 0, "total_frames": 0, "percentage": 0},
            "frame_queue": fq,
            "result":      None,
            "error":       None,
        }

    async def run():
        try:
            def frame_cb(fidx, total, jpeg_bytes, fps, dets, mines):
                data = {
                    "frame":      fidx,
                    "total":      total,
                    "fps":        round(fps, 1),
                    "detections": dets,
                    "mines":      mines,
                    "img":        base64.b64encode(jpeg_bytes).decode(),
                }
                # Block if queue is full — back-pressures the pipeline
                # rather than dropping frames (smoother 30 FPS playback).
                fq.put(data)

            def progress_cb(current, total):
                with processing_lock:
                    if job_id in processing_state:
                        p = processing_state[job_id]["progress"]
                        p["current_frame"] = current
                        p["total_frames"]  = total
                        p["percentage"]    = int(current/total*100) if total else 0

            loop   = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: process_video(
                    str(input_path), str(output_path),
                    progress_callback=progress_cb,
                    frame_callback=frame_cb,
                    yolo_enabled=request.yolo_enabled,
                    hud_enabled=request.hud_enabled,
                ),
            )
            with processing_lock:
                processing_state[job_id]["status"] = "completed"
                processing_state[job_id]["result"] = result
                processing_state[job_id]["progress"]["percentage"] = 100
            fq.put(None)   # sentinel → SSE generator exits cleanly
            logger.info(f"✅ Done: {output_filename}")

        except Exception as exc:
            logger.error(f"Processing failed: {exc}", exc_info=True)
            with processing_lock:
                if job_id in processing_state:
                    processing_state[job_id]["status"] = "failed"
                    processing_state[job_id]["error"]  = str(exc)
            fq.put(None)

    background_tasks.add_task(run)

    return {
        "status":        "processing_started",
        "job_id":        job_id,
        "output_file":   output_filename,
        "source_fps":    round(source_fps, 3),
        "source_frames": source_frames,
    }

# ─────────────────────────────────────────────
# SSE frame stream
# ─────────────────────────────────────────────
@app.get("/api/stream/{job_id}")
async def api_stream(job_id: str):
    """
    Server-Sent Events endpoint.
    Streams processed frames as base64 JPEG until the job is done.
    """
    async def generator():
        # Wait up to 8 s for the job to appear
        for _ in range(80):
            with processing_lock:
                if job_id in processing_state:
                    break
            await asyncio.sleep(0.1)
        else:
            yield f"data: {json.dumps({'error': 'job not found'})}\n\n"
            return

        loop = asyncio.get_event_loop()

        with processing_lock:
            fq = processing_state[job_id]["frame_queue"]

        while True:
            try:
                # Block in executor with 2 s timeout
                item = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: fq.get(timeout=2.0)),
                    timeout=3.0,
                )
                if item is None:
                    # Job finished
                    with processing_lock:
                        state = processing_state.get(job_id, {})
                    yield f"data: {json.dumps({'done': True, 'status': state.get('status'), 'result': state.get('result')})}\n\n"
                    return
                yield f"data: {json.dumps(item)}\n\n"

            except (queue.Empty, asyncio.TimeoutError):
                # Check if job has ended without sentinel (shouldn't happen, safety net)
                with processing_lock:
                    status = processing_state.get(job_id, {}).get("status", "")
                if status in ("completed", "failed"):
                    yield f"data: {json.dumps({'done': True, 'status': status})}\n\n"
                    return
                # Keepalive comment so the connection stays open
                yield ": keepalive\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering":"no",
            "Connection":       "keep-alive",
        },
    )

# ─────────────────────────────────────────────
# Progress poll (for clients that don't use SSE)
# ─────────────────────────────────────────────
@app.get("/api/progress/{job_id}")
async def api_progress(job_id: str):
    with processing_lock:
        if job_id not in processing_state:
            raise HTTPException(404, "Job not found")
        state = dict(processing_state[job_id])
        state.pop("frame_queue", None)   # not JSON-serialisable
    return state

# ─────────────────────────────────────────────
# Image processing  (synchronous, returns b64)
# ─────────────────────────────────────────────
@app.post("/api/process-image")
async def api_process_image(
    file:         UploadFile = File(...),
    yolo_enabled: bool = Query(True),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE:
        raise HTTPException(400, f"Not a supported image type: {ext}")

    data = await file.read()
    tmp_in  = UPLOAD_DIR  / file.filename
    tmp_out = OUTPUT_DIR  / f"processed_{Path(file.filename).stem}.jpg"
    tmp_in.write_bytes(data)

    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: process_image(str(tmp_in), str(tmp_out), yolo_enabled=yolo_enabled),
        )
    except Exception as exc:
        raise HTTPException(500, f"Image processing failed: {exc}")

    return {"status": "success", **result}

# ─────────────────────────────────────────────
# Download processed file
# ─────────────────────────────────────────────
@app.get("/api/download/{filename}")
async def api_download(filename: str):
    p = OUTPUT_DIR / filename
    if not p.exists():
        raise HTTPException(404, f"File not found: {filename}")
    media = "video/mp4" if filename.endswith(".mp4") else "application/octet-stream"
    return FileResponse(path=p, media_type=media, filename=filename)

# ─────────────────────────────────────────────
# Misc
# ─────────────────────────────────────────────
@app.get("/api/list-outputs")
async def api_list_outputs():
    files = [
        {"filename": f.name, "size_mb": round(f.stat().st_size/(1024*1024), 2)}
        for f in OUTPUT_DIR.glob("*") if f.is_file()
    ]
    return {"files": files, "count": len(files)}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "AQUAVISION", "version": "2.0.0"}


@app.post("/api/warmup")
async def api_warmup(background_tasks: BackgroundTasks):
    """
    FIX 3: Pre-load TRT engines in the background so the very first
    video processed has no model-loading delay.
    Called automatically when the upload page loads.
    """
    async def _load():
        try:
            from inference_module import _load_models
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _load_models)
            logger.info("✓ Models pre-warmed via /api/warmup")
        except Exception as exc:
            logger.warning(f"Warmup non-critical error: {exc}")
    background_tasks.add_task(_load)
    return {"status": "warming_up"}

@app.get("/api/status")
async def status():
    with processing_lock:
        active = sum(1 for s in processing_state.values() if s["status"] == "processing")
    return {
        "status":          "running",
        "active_jobs":     active,
        "uploaded_videos": len(list(UPLOAD_DIR.glob("*"))),
        "processed_files": len(list(OUTPUT_DIR.glob("*"))),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
