import sys
import os

# Fix import paths for Docker
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from fastapi import FastAPI, HTTPException
from tasks import load_task
from models import Action

# 🔥 Import OpenAI (but DO NOT initialize globally)
from openai import OpenAI

app = FastAPI(title="Feature Engineering RL Agent")

current_env = None


# ✅ Health Check
@app.get("/")
def health_check():
    return {
        "status": "online",
        "message": "OpenEnv Feature Engineering is ready."
    }


# ✅ Reset Environment
@app.post("/reset")
def reset(task_id: str = "easy"):
    global current_env
    try:
        current_env = load_task(task_id)
        return current_env.reset()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ✅ Step Execution (LLM SAFE INTEGRATION)
@app.post("/step")
def step(action: Action):
    global current_env

    if current_env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )

    # 🔥 Safe LLM call (won’t crash if env vars missing)
    try:
        api_key = os.environ.get("API_KEY")
        base_url = os.environ.get("API_BASE_URL")

        # Only call LLM if env variables exist
        if api_key and base_url:
            client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )

            response = client.chat.completions.create(
                model=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in feature engineering."
                    },
                    {
                        "role": "user",
                        "content": f"Suggest a preprocessing step for: {action}"
                    }
                ]
            )

            # Optional logging
            print("LLM Output:", response.choices[0].message.content)

    except Exception as e:
        # Never crash the app because of LLM
        print("LLM skipped or failed:", str(e))

    # ✅ Continue RL step
    result = current_env.step(action)
    return result


# ✅ Get Current State
@app.get("/state")
def get_state():
    global current_env

    if current_env is None:
        return {
            "error": "No active environment",
            "step_count": 0
        }

    return {
        "observation": current_env.get_state(),
        "step_count": getattr(current_env, 'steps', 0),
        "task": "titanic"
    }


# ✅ Main entry point
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
