import os
import requests
from openai import OpenAI

# Required Environment Variables per Hackathon Rules
API_BASE_URL = os.getenv("API_BASE_URL", "https://sargam-ghagre-feature-engineer-rl.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo") # Fallback for local testing
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize OpenAI-compatible client (Pointed to your Space or Proxy)
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def run_baseline():
    print("--- Starting Baseline Inference ---")
    
    # 1. Reset Environment
    # We use requests here because this is the environment setup
    print(f"Step 1: Resetting to 'easy' task...")
    reset_url = f"{API_BASE_URL}/reset?task_id=easy"
    reset_req = requests.post(reset_url)
    
    if reset_req.status_code == 200:
        initial_score = reset_req.json()['observation']['current_score']
        print(f"Success! Initial Accuracy: {initial_score:.4f}")
    
    # 2. Take a Step (Example Action)
    print("\nStep 2: Performing Mean Imputation on 'feature1'...")
    action = {"action_type": "impute_mean", "column": "feature1"}
    step_req = requests.post(f"{API_BASE_URL}/step", json=action)
    
    if step_req.status_code == 200:
        result = step_req.json()
        print(f"Action Complete! Reward Received: {result['reward']:.4f}")
    else:
        print(f"Error: {step_req.text}")

if __name__ == "__main__":
    run_baseline()
