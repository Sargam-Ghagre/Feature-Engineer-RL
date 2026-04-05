import sys
import os

# The "Absolute Path Fix" for Docker
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from fastapi import FastAPI, HTTPException
from tasks import load_task
from models import Action
from typing import Optional

app = FastAPI(title="Feature Engineering RL Agent")
current_env = None

@app.get("/")
def health_check():
    return {"status": "online", "message": "OpenEnv Feature Engineering is ready."}

@app.post("/reset")
def reset(task_id: str = "easy"):
    global current_env
    try:
        current_env = load_task(task_id)
        return current_env.reset() 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
def step(action: Action):
    global current_env
    if current_env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    result = current_env.step(action)
    return result

@app.get("/state")
def get_state():
    global current_env
    if current_env is None:
        return {"error": "No active environment", "step_count": 0}
    
    return {
        "observation": current_env.get_state(),
        "step_count": getattr(current_env, 'steps', 0),
        "task": "titanic"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
