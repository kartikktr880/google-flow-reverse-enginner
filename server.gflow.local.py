import asyncio
import os
import uuid
import subprocess
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

OUTPUT_DIR = Path(r"C:\Users\Administrator\google-flow\output\scenes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Studio Video Production Engine API", version="2.0.0")

TASKS = {}
job_queue = asyncio.Queue()

class SceneRequest(BaseModel):
    prompt: str
    duration_seconds: int = 5
    aspect_ratio: str = "9:16"
    reference_image_path: Optional[str] = None
    character_name: Optional[str] = None

class CharacterRequest(BaseModel):
    name: str
    description: str

def extract_last_frame(video_path: str, output_image_path: str):
    """Extracts the last frame of a video using ffmpeg for scene continuation."""
    try:
        cmd = [
            "ffmpeg", "-y", "-sseof", "-0.1", "-i", video_path,
            "-update", "1", "-q:v", "2", output_image_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return output_image_path
    except Exception:
        return None

async def studio_worker():
    while True:
        task_id = await job_queue.get()
        task = TASKS.get(task_id)
        if not task:
            job_queue.task_done()
            continue

        task["status"] = "IN_PROGRESS"
        
        # Build command: Use r2v/i2v if reference image exists, otherwise t2v
        ref_path = task.get("reference_image_path")
        if ref_path and os.path.exists(ref_path):
            cmd = [
                "gflow", "video", "r2v",
                task["prompt"],
                "--ref", str(ref_path),
                "--duration", str(task["duration_seconds"]),
                "--model", "veo-3.1-fast",
                "--out-dir", str(OUTPUT_DIR)
            ]
        else:
            cmd = [
                "gflow", "video", "t2v",
                task["prompt"],
                "--duration", str(task["duration_seconds"]),
                "--model", "veo-3.1-fast",
                "--out-dir", str(OUTPUT_DIR)
            ]
            
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                task["status"] = "COMPLETED"
                # Find latest mp4 generated
                mp4_files = sorted(OUTPUT_DIR.glob("*.mp4"), key=os.path.getmtime, reverse=True)
                if mp4_files:
                    latest_video = mp4_files[0]
                    task["video_path"] = str(latest_video)
                    task["download_url"] = f"/api/v1/assets/{latest_video.name}"
                    
                    # Extract last frame for seamless scene continuation
                    frame_path = str(OUTPUT_DIR / f"frame_{latest_video.stem}.jpg")
                    last_frame = extract_last_frame(str(latest_video), frame_path)
                    task["last_frame_path"] = last_frame
            else:
                task["status"] = "FAILED"
                task["error"] = stderr.decode(errors="replace").strip()
        except Exception as e:
            task["status"] = "FAILED"
            task["error"] = str(e)

        job_queue.task_done()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(studio_worker())

@app.get("/api/v1/health")
def health():
    return {"status": "operational", "engine": "studio-video-core", "port": 8080}

@app.post("/api/v1/characters/create")
def create_character(req: CharacterRequest):
    """Locks a character identity descriptor for the session."""
    return {
        "character_id": str(uuid.uuid4())[:8],
        "name": req.name,
        "anchor_prompt": req.description,
        "status": "READY"
    }

@app.post("/api/v1/scenes/generate", status_code=202)
async def generate_scene(req: SceneRequest):
    # Enforce duration bounds between 4 and 10 seconds
    clamped_duration = max(4, min(10, req.duration_seconds))
    task_id = f"task_{uuid.uuid4().hex[:10]}"
    
    TASKS[task_id] = {
        "task_id": task_id,
        "prompt": req.prompt,
        "duration_seconds": clamped_duration,
        "aspect_ratio": req.aspect_ratio,
        "reference_image_path": req.reference_image_path,
        "status": "QUEUED",
        "video_path": None,
        "download_url": None,
        "last_frame_path": None
    }
    await job_queue.put(task_id)
    return {"task_id": task_id, "status": "QUEUED", "duration": clamped_duration}

@app.get("/api/v1/tasks/{task_id}")
def check_status(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.get("/api/v1/assets/{filename}")
def download_asset(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(path=path, media_type="video/mp4", filename=filename)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
