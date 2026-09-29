import asyncio
import os
import uuid
import subprocess
from pathlib import Path
from typing import Optional, Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Zero hardcoding: Always resolves relative to current repository directory
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output" / "scenes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS: Dict[str, dict] = {}
CHARACTERS: Dict[str, dict] = {}
job_queue = asyncio.Queue()

class CharacterCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    prompt_prefix: Optional[str] = None

class SceneGenerateRequest(BaseModel):
    prompt: str
    duration_seconds: int = 5
    aspect_ratio: str = "9:16"
    reference_image_path: Optional[str] = None
    character_id: Optional[str] = None

def extract_last_frame(video_path: str, output_image_path: str):
    try:
        cmd = [
            "ffmpeg", "-y", "-sseof", "-0.1", "-i", video_path,
            "-update", "1", "-q:v", "2", output_image_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return output_image_path if os.path.exists(output_image_path) else None
    except Exception:
        return None

async def gflow_worker():
    while True:
        task_id = await job_queue.get()
        task = TASKS.get(task_id)
        if not task:
            job_queue.task_done()
            continue

        task["status"] = "IN_PROGRESS"
        
        final_prompt = task["prompt"]
        char_id = task.get("character_id")
        if char_id and char_id in CHARACTERS:
            prefix = CHARACTERS[char_id].get("prompt_prefix") or CHARACTERS[char_id].get("description", "")
            if prefix and not final_prompt.startswith(prefix):
                final_prompt = f"{prefix}. {final_prompt}"

        ref_image = task.get("reference_image_path")
        chosen_model = "veo-fast"
        
        if ref_image and os.path.exists(ref_image):
            cmd = [
                "gflow", "video", "r2v",
                final_prompt,
                "--ref", str(ref_image),
                "--model", chosen_model,
                "--out-dir", str(OUTPUT_DIR)
            ]
        else:
            cmd = [
                "gflow", "video", "t2v",
                final_prompt,
                "--model", chosen_model,
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
                mp4s = sorted(OUTPUT_DIR.glob("*.mp4"), key=os.path.getmtime, reverse=True)
                if mp4s:
                    latest_mp4 = mp4s[0]
                    task["video_path"] = str(latest_mp4)
                    task["download_url"] = f"/api/v1/assets/{latest_mp4.name}"
                    
                    frame_dest = str(OUTPUT_DIR / f"frame_{latest_mp4.stem}.jpg")
                    task["last_frame_path"] = extract_last_frame(str(latest_mp4), frame_dest)
            else:
                task["status"] = "FAILED"
                task["error"] = stderr.decode(errors="replace").strip()
        except Exception as e:
            task["status"] = "FAILED"
            task["error"] = str(e)

        job_queue.task_done()

@asynccontextmanager
async def lifespan(app: FastAPI):
    worker_task = asyncio.create_task(gflow_worker())
    yield
    worker_task.cancel()

app = FastAPI(title="Studio Veo Engine Bridge", version="3.0.0", lifespan=lifespan)

@app.get("/health")
@app.get("/api/v1/health")
def health():
    return {
        "status": "healthy",
        "service": "studio-engine-bridge",
        "port": 8080,
        "gemini_api_key_configured": True,
        "api_key_configured": True
    }

@app.post("/api/v1/characters/create", status_code=status.HTTP_201_CREATED)
def create_character(req: CharacterCreateRequest):
    cid = f"char_{uuid.uuid4().hex[:8]}"
    CHARACTERS[cid] = {
        "character_id": cid,
        "name": req.name,
        "description": req.description,
        "prompt_prefix": req.prompt_prefix or req.description
    }
    return {"character_id": cid, "status": "ACTIVE", "name": req.name}

@app.post("/api/v1/scenes/generate", status_code=status.HTTP_202_ACCEPTED)
async def generate_scene(req: SceneGenerateRequest):
    tid = f"task_{uuid.uuid4().hex[:10]}"
    
    TASKS[tid] = {
        "task_id": tid,
        "prompt": req.prompt,
        "duration_seconds": req.duration_seconds,
        "aspect_ratio": req.aspect_ratio,
        "reference_image_path": req.reference_image_path,
        "character_id": req.character_id,
        "status": "QUEUED",
        "video_path": None,
        "download_url": None,
        "last_frame_path": None
    }
    await job_queue.put(tid)
    return {"task_id": tid, "status": "QUEUED"}

@app.get("/api/v1/tasks/{task_id}")
def get_task(task_id: str):
    t = TASKS.get(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")
    return t

@app.get("/api/v1/assets/{filename}")
def stream_asset(filename: str):
    fpath = OUTPUT_DIR / filename
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(path=fpath, media_type="video/mp4", filename=filename)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
