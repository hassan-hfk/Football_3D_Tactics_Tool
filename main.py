"""
main.py — FastAPI server for football 3D tactics tool.

Endpoints:
  GET  /                       — serves the Forma B frontend
  POST /process                — upload video, start pipeline, returns job_id
  GET  /status/{id}            — poll pipeline progress
  GET  /result/{id}            — get pipeline result JSON when done
  POST /process-mesh/{job_id}  — start GVHMR mesh lift for a finished job
  GET  /status-mesh/{id}       — poll mesh lift progress
  GET  /result-mesh/{id}       — stream the _mesh.json when done
  GET  /logs                   — SSE stream of all server log lines (terminal)
  POST /export/{job_id}        — start server-side 2D overlay video export
  GET  /status-export/{id}     — poll export progress
  GET  /download-export/{id}/{stage} — download one of the five 2D overlays
  POST /transcode-to-mp4       — accept a WebM blob from the browser, return MP4

Run locally:
  pip install -r requirements.txt
  python main.py

Then open: http://localhost:8000
"""

import os
import sys
import uuid
import json
import queue
import asyncio
import threading
import traceback
import subprocess
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import uvicorn

from pipeline import process_video, process_image


# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="Football 3D Tactics")

UPLOAD_DIR   = "uploads"
OUTPUT_DIR   = "outputs"
TRANSCODE_DIR = os.path.join(OUTPUT_DIR, "transcode")   # temp WebM/MP4 staging
os.makedirs(UPLOAD_DIR,    exist_ok=True)
os.makedirs(OUTPUT_DIR,    exist_ok=True)
os.makedirs(TRANSCODE_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

jobs: dict = {}


# ─── Log broadcast ────────────────────────────────────────────────────────────
# Every print() / stderr write in this process (including from pipeline.py,
# lift_gvhmr.py, ultralytics, etc.) is intercepted and broadcast to all
# connected SSE clients so the in-browser terminal stays up to date.

_log_subscribers: list[queue.Queue] = []
_log_lock = threading.Lock()
_MAX_LOG_LINES = 2000          # ring-buffer kept in memory for late joiners
_log_history: list[str] = []


def _broadcast(line: str):
    """Push one log line to every connected SSE client and to history."""
    global _log_history
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {line}"
    with _log_lock:
        _log_history.append(entry)
        if len(_log_history) > _MAX_LOG_LINES:
            _log_history = _log_history[-_MAX_LOG_LINES:]
        for q in list(_log_subscribers):
            try:
                q.put_nowait(entry)
            except queue.Full:
                pass


class _Tee:
    """Wraps an existing stream (stdout/stderr), forwards to it AND broadcasts."""
    def __init__(self, original):
        self._orig = original

    def write(self, text):
        self._orig.write(text)
        if text and text.strip():
            for line in text.splitlines():
                if line.strip():
                    _broadcast(line)

    def flush(self):
        self._orig.flush()

    # Delegate everything else so logging handlers stay happy
    def __getattr__(self, name):
        return getattr(self._orig, name)


# Install tees immediately so every subsequent print() is captured
sys.stdout = _Tee(sys.__stdout__)
sys.stderr = _Tee(sys.__stderr__)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("static/FORMA B.dc.html")


@app.get("/logs")
async def logs():
    """
    Server-Sent Events endpoint. The browser connects once and receives every
    log line as  `data: <line>\n\n`.  History (up to _MAX_LOG_LINES) is
    replayed first so the terminal shows what happened before the tab opened.
    """
    q: queue.Queue = queue.Queue(maxsize=500)

    async def event_stream():
        with _log_lock:
            history_snapshot = list(_log_history)
            _log_subscribers.append(q)

        try:
            for line in history_snapshot:
                yield f"data: {line}\n\n"

            loop = asyncio.get_event_loop()
            while True:
                try:
                    line = await loop.run_in_executor(None, q.get, True, 1.0)
                    yield f"data: {line}\n\n"
                except queue.Empty:
                    yield ": ping\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            with _log_lock:
                try:
                    _log_subscribers.remove(q)
                except ValueError:
                    pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# Accepted upload types. Images run the same detection/homography/team/pose
# pipeline as video and emit a one-frame result; the mesh stage handles them
# too (it replicates the still into a short static clip for GVHMR).
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@app.post("/process")
async def process(video: UploadFile = File(...)):
    """Upload a video OR a still image and start the pipeline. Returns job_id
    to poll. Images emit a one-frame result with the same schema."""
    name = (video.filename or "").lower()
    is_image = name.endswith(IMAGE_EXTS)
    if not (name.endswith(VIDEO_EXTS) or is_image):
        raise HTTPException(
            400,
            "Unsupported file type. Use a video (mp4, avi, mov, mkv) "
            "or an image (jpg, jpeg, png, bmp, webp)."
        )

    job_id     = str(uuid.uuid4())[:8]
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{video.filename}")

    content = await video.read()
    with open(input_path, "wb") as f:
        f.write(content)

    file_mb = len(content) / 1024 / 1024
    kind = "image" if is_image else "video"
    print(f"[pipeline] New {kind} job {job_id} — {video.filename} ({file_mb:.1f} MB)")

    jobs[job_id] = {
        "status"    : "processing",
        "progress"  : {"frame": 0, "total": 0, "percent": 0},
        "error"     : None,
        "video_path": input_path,   # key kept for downstream mesh/export reuse
        "is_image"  : is_image,
    }

    threading.Thread(
        target=_run_pipeline,
        args=(job_id, input_path, is_image),
        daemon=True,
    ).start()

    return {"job_id": job_id, "filename": video.filename, "kind": kind}


@app.get("/status/{job_id}")
async def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    return {"status": job["status"], "progress": job["progress"], "error": job["error"]}


@app.get("/result/{job_id}")
async def result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] == "processing":
        raise HTTPException(400, "Still processing — poll /status first")
    if job["status"] == "error":
        raise HTTPException(500, f"Processing failed: {job['error']}")
    result_path = os.path.join(OUTPUT_DIR, f"{job_id}.json")
    if not os.path.exists(result_path):
        raise HTTPException(500, "Result file missing")
    return FileResponse(result_path, media_type="application/json")


# ─── Mesh lift ────────────────────────────────────────────────────────────────

@app.post("/process-mesh/{job_id}")
async def process_mesh(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, f"Pipeline job '{job_id}' not found")
    pipeline_job = jobs[job_id]
    if pipeline_job["status"] != "done":
        raise HTTPException(400, f"Pipeline job '{job_id}' not done (status: {pipeline_job['status']})")

    result_json = os.path.join(OUTPUT_DIR, f"{job_id}.json")
    if not os.path.exists(result_json):
        raise HTTPException(500, f"Result JSON for job '{job_id}' not found on disk")

    mesh_job_id = f"{job_id}_mesh"
    existing = jobs.get(mesh_job_id)
    if existing and existing["status"] in ("processing", "done"):
        return {"mesh_job_id": mesh_job_id, "status": existing["status"]}

    video_path = pipeline_job.get("video_path")
    if video_path and not os.path.exists(video_path):
        video_path = _discover_video(job_id)

    jobs[mesh_job_id] = {
        "status"  : "processing",
        "progress": {"stage": "queued", "percent": 0},
        "error"   : None,
    }

    threading.Thread(
        target=_run_mesh_lift,
        args=(mesh_job_id, result_json, video_path),
        daemon=True,
    ).start()

    return {"mesh_job_id": mesh_job_id, "status": "processing"}


@app.get("/status-mesh/{mesh_job_id}")
async def status_mesh(mesh_job_id: str):
    if mesh_job_id not in jobs:
        raise HTTPException(404, "Mesh job not found")
    job = jobs[mesh_job_id]
    return {"status": job["status"], "progress": job["progress"], "error": job["error"]}


@app.get("/result-mesh/{mesh_job_id}")
async def result_mesh(mesh_job_id: str):
    if mesh_job_id not in jobs:
        raise HTTPException(404, "Mesh job not found")
    job = jobs[mesh_job_id]
    if job["status"] == "processing":
        raise HTTPException(400, "Mesh lift still running — poll /status-mesh first")
    if job["status"] == "error":
        raise HTTPException(500, f"Mesh lift failed: {job['error']}")
    base_job_id = mesh_job_id[: -len("_mesh")]
    mesh_path   = os.path.join(OUTPUT_DIR, f"{base_job_id}_mesh.json")
    if not os.path.exists(mesh_path):
        raise HTTPException(500, "Mesh JSON missing on disk")
    return FileResponse(mesh_path, media_type="application/json")


# ─── Background runners ───────────────────────────────────────────────────────

def _run_pipeline(job_id: str, input_path: str, is_image: bool = False):
    try:
        print(f"[pipeline:{job_id}] Starting {'image ' if is_image else ''}pipeline…")
        jobs[job_id]["progress"] = {"frame": 0, "total": 0, "percent": 1, "stage": "Loading models…"}

        def on_progress(p):
            if job_id in jobs:
                jobs[job_id]["progress"] = p
            if p.get("frame", 0) % 60 == 0 and p.get("total", 0) > 0:
                pct = p.get("percent", 0)
                print(f"[pipeline:{job_id}] Frame {p['frame']}/{p['total']}  ({pct:.0f}%)")

        if is_image:
            data = process_image(input_path, on_progress=on_progress)
        else:
            data = process_video(input_path, on_progress=on_progress)

        result_path = os.path.join(OUTPUT_DIR, f"{job_id}.json")
        with open(result_path, "w") as f:
            json.dump(data, f)

        jobs[job_id]["status"] = "done"
        jobs[job_id]["progress"]["percent"] = 100
        n = data["metadata"]["processed_frames"]
        print(f"[pipeline:{job_id}] ✓ Done — {n} frames processed")

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)
        print(f"[pipeline:{job_id}] ✗ FAILED: {e}")
        traceback.print_exc()


def _run_mesh_lift(mesh_job_id: str, result_json: str, video_path: str):
    try:
        print(f"[gvhmr:{mesh_job_id}] Loading models…")
        jobs[mesh_job_id]["progress"] = {"stage": "Loading models…", "percent": 3, "track": 0, "total_tracks": 0}

        import lift_gvhmr

        print(f"[gvhmr:{mesh_job_id}] Running GVHMR lift…")
        jobs[mesh_job_id]["progress"] = {"stage": "Scanning tracks…", "percent": 8, "track": 0, "total_tracks": 0}

        def on_mesh_progress(done: int, total: int, stage: str = ""):
            """Called from lift_gvhmr after each segment is processed."""
            if total > 0:
                # Reserve 8-98% for actual lifting; loading took 0-8%.
                pct = round(8 + (done / total) * 90, 1)
            else:
                pct = 8
            jobs[mesh_job_id]["progress"] = {
                "stage": stage or f"Track {done}/{total}",
                "percent": pct,
                "track": done,
                "total_tracks": total,
            }

        lift_gvhmr.process(
            result_path=result_json,
            video_path=video_path,
            verts_dtype="fp16",
            on_progress=on_mesh_progress,
        )

        jobs[mesh_job_id]["status"] = "done"
        jobs[mesh_job_id]["progress"] = {"stage": "done", "percent": 100}
        print(f"[gvhmr:{mesh_job_id}] ✓ Done")

    except Exception as e:
        jobs[mesh_job_id]["status"] = "error"
        jobs[mesh_job_id]["error"]  = str(e)
        print(f"[gvhmr:{mesh_job_id}] ✗ FAILED: {e}")
        traceback.print_exc()

    finally:
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"[gvhmr:{mesh_job_id}] Cleaned up upload: {video_path}")
            except OSError:
                pass


# ─── 2D overlay video export (existing) ───────────────────────────────────────

@app.post("/export/{job_id}")
async def export_videos(job_id: str):
    """
    Start server-side OpenCV video export for a finished pipeline job.
    Returns immediately; poll /status-export/{job_id} for progress.
    """
    if job_id not in jobs:
        raise HTTPException(404, f"Job '{job_id}' not found")
    if jobs[job_id]["status"] != "done":
        raise HTTPException(400, f"Job '{job_id}' not done yet")

    exp_id = f"{job_id}_export"
    existing = jobs.get(exp_id)
    if existing and existing["status"] in ("processing", "done"):
        return {"export_id": exp_id, "status": existing["status"]}

    result_json = os.path.join(OUTPUT_DIR, f"{job_id}.json")
    video_path  = jobs[job_id].get("video_path") or _discover_video(job_id)

    with open(result_json) as f:
        meta = json.load(f).get("metadata", {})
    kc = (meta.get("teams") or {}).get("kit_colors", ["#CC2222", "#1144CC"])
    team_hex = (kc[0] if len(kc) > 0 else "#CC2222",
                kc[1] if len(kc) > 1 else "#1144CC")

    jobs[exp_id] = {
        "status"  : "processing",
        "progress": {"stage": "", "done": 0, "total": 0},
        "error"   : None,
    }

    def _run():
        try:
            import video_export

            def on_progress(stage, done, total):
                jobs[exp_id]["progress"] = {
                    "stage": stage, "done": done, "total": total,
                    "percent": round(done / max(total, 1) * 100, 1)
                }

            video_export.export_all(
                result_json_path=result_json,
                video_path=video_path,
                output_dir=OUTPUT_DIR,
                on_progress=on_progress,
                team_hex=team_hex,
            )
            jobs[exp_id]["status"] = "done"
            jobs[exp_id]["progress"]["percent"] = 100
            print(f"[export] Job {exp_id} complete")
        except Exception as e:
            jobs[exp_id]["status"] = "error"
            jobs[exp_id]["error"]  = str(e)
            print(f"[export] Job {exp_id} failed: {e}")
            traceback.print_exc()

    threading.Thread(target=_run, daemon=True).start()
    return {"export_id": exp_id, "status": "processing"}


@app.get("/status-export/{exp_id}")
async def status_export(exp_id: str):
    if exp_id not in jobs:
        raise HTTPException(404, "Export job not found")
    j = jobs[exp_id]
    return {"status": j["status"], "progress": j["progress"], "error": j["error"]}


@app.get("/download-export/{job_id}/{stage}")
async def download_export(job_id: str, stage: str):
    """Stream a finished export video for one pipeline stage."""
    allowed = {"detection", "clustering", "keypoints", "pnlcalib", "pose"}
    if stage not in allowed:
        raise HTTPException(400, f"Unknown stage '{stage}'")
    path = os.path.join(OUTPUT_DIR, f"{job_id}_{stage}.mp4")
    if not os.path.exists(path):
        raise HTTPException(404, f"Export video not found: {path}")
    return FileResponse(path, media_type="video/mp4",
                        filename=f"forma_{job_id}_{stage}.mp4")


# ─── 3D mesh viewer MP4 export (NEW) ──────────────────────────────────────────
# The mesh viewer (static/mesh_viewer.html) records the WebGL canvas as a
# WebM blob via MediaRecorder, then POSTs it here. We transcode it to MP4
# using the ffmpeg binary bundled with imageio-ffmpeg (already a transitive
# dependency through ultralytics' requirements), so no system-wide ffmpeg
# install is needed. The transcoded MP4 is streamed back as the response.

_FFMPEG_EXE_CACHE: str | None = None


def _find_ffmpeg() -> str | None:
    """Locate an ffmpeg binary. Prefers imageio_ffmpeg's bundled copy
    (works on Windows, macOS, Linux without any system install), falls back
    to whatever is on PATH. Result is cached so repeated calls are cheap."""
    global _FFMPEG_EXE_CACHE
    if _FFMPEG_EXE_CACHE is not None:
        return _FFMPEG_EXE_CACHE or None
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            _FFMPEG_EXE_CACHE = exe
            return exe
    except Exception:
        pass
    # PATH fallback
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        _FFMPEG_EXE_CACHE = exe
        return exe
    _FFMPEG_EXE_CACHE = ""    # sentinel: looked, didn't find
    return None


@app.post("/transcode-to-mp4")
async def transcode_to_mp4(
    video: UploadFile = File(...),
    mime_type: str = Form(default="video/webm"),
):
    """
    Accept a browser-recorded WebM (or any container ffmpeg can read), encode
    to MP4 (H.264 + AAC if audio is present), return the MP4 file.

    Pipeline:
      1. Save the upload to a temp file in TRANSCODE_DIR.
      2. Run ffmpeg with libx264, yuv420p (universal pixel format),
         +faststart (so the MP4 is playable while still downloading).
      3. Stream the MP4 back. Clean up both temp files in the background.
    """
    ffmpeg_exe = _find_ffmpeg()
    if not ffmpeg_exe:
        raise HTTPException(
            500,
            "ffmpeg not found. Install imageio-ffmpeg (pip install imageio-ffmpeg) "
            "or add ffmpeg to PATH."
        )

    content = await video.read()
    if not content:
        raise HTTPException(400, "Empty upload")
    size_mb = len(content) / 1024 / 1024
    print(f"[transcode] Received {size_mb:.1f} MB ({mime_type})")

    # 250 MB hard cap — a 90s 1080p WebM at 6 Mbps is roughly 70 MB, so this
    # is generous. Keeps a runaway upload from filling the disk.
    if len(content) > 250 * 1024 * 1024:
        raise HTTPException(413, f"Upload too large: {size_mb:.1f} MB (max 250 MB)")

    transcode_id = str(uuid.uuid4())[:8]
    # Pick the input extension from the mime type when we can; default .webm.
    if "mp4" in mime_type.lower():
        in_ext = ".mp4"
    elif "mov" in mime_type.lower() or "quicktime" in mime_type.lower():
        in_ext = ".mov"
    else:
        in_ext = ".webm"

    in_path  = os.path.join(TRANSCODE_DIR, f"{transcode_id}_in{in_ext}")
    out_path = os.path.join(TRANSCODE_DIR, f"{transcode_id}_tactical.mp4")

    try:
        with open(in_path, "wb") as f:
            f.write(content)

        # ── ffmpeg invocation ─────────────────────────────────────────────
        # -i           : input file
        # -c:v libx264 : H.264 (universally compatible with QuickTime, web,
        #                phones, social media uploads)
        # -preset fast : ~3x faster than 'medium', minor file size cost
        # -crf 22      : visually lossless quality at sensible bitrate
        # -pix_fmt yuv420p : required for QuickTime/iOS playback
        # -movflags +faststart : moov atom moved to file start; lets the
        #                       browser begin playback before download is
        #                       complete
        # -an          : drop audio (canvas captureStream has none, but if
        #                the source ever has audio this keeps file size down).
        #                Remove -an if you ever want audio.
        cmd = [
            ffmpeg_exe,
            "-y",                       # overwrite output if exists
            "-i", in_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            out_path,
        ]
        print(f"[transcode] ffmpeg: {' '.join(cmd)}")

        # Run synchronously on the worker thread FastAPI assigns to def
        # endpoints. For a 60s clip libx264 -preset fast usually finishes
        # in 5-15 s on a modern CPU, so we keep the request handler simple
        # rather than introducing a job-id polling dance.
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
        )

        if result.returncode != 0:
            err_tail = (result.stderr or "")[-800:].strip()
            print(f"[transcode] ✗ ffmpeg failed (exit {result.returncode})\n{err_tail}")
            raise HTTPException(
                500,
                f"ffmpeg failed (exit {result.returncode}). "
                f"Tail of stderr: {err_tail[-400:]}"
            )

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise HTTPException(500, "ffmpeg produced no output")

        mp4_size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"[transcode] ✓ {transcode_id}: {size_mb:.1f}MB → {mp4_size_mb:.1f}MB MP4")

    except subprocess.TimeoutExpired:
        # Best-effort cleanup before bubbling up
        for p in (in_path, out_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
        raise HTTPException(504, "ffmpeg timed out (10 minute limit)")
    except HTTPException:
        # Clean up input on any error path; output may or may not exist
        try:
            if os.path.exists(in_path):
                os.remove(in_path)
        except OSError:
            pass
        raise
    except Exception as e:
        try:
            if os.path.exists(in_path):
                os.remove(in_path)
            if os.path.exists(out_path):
                os.remove(out_path)
        except OSError:
            pass
        print(f"[transcode] ✗ unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(500, f"Transcode failed: {e}")

    # Drop the input file now — the output will be cleaned up after the
    # FileResponse finishes streaming via a BackgroundTask.
    try:
        os.remove(in_path)
    except OSError:
        pass

    from starlette.background import BackgroundTask

    def _cleanup_output():
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except OSError:
            pass

    return FileResponse(
        out_path,
        media_type="video/mp4",
        filename="tactical_analysis.mp4",
        background=BackgroundTask(_cleanup_output),
    )


def _discover_video(job_id: str):
    import glob
    for ext in (".mp4", ".mov", ".mkv", ".avi",
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        matches = glob.glob(os.path.join(UPLOAD_DIR, f"{job_id}_*{ext}"))
        if matches:
            return matches[0]
    return None


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[server] Starting Football 3D Tactics server (Forma B)…")
    print("[server] Open http://localhost:8000 in your browser")
    # Check ffmpeg availability up front so users know before recording.
    _ff = _find_ffmpeg()
    if _ff:
        print(f"[server] ffmpeg: {_ff}")
    else:
        print("[server] WARNING: ffmpeg not found — MP4 export will fail. "
              "Run: pip install imageio-ffmpeg")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)