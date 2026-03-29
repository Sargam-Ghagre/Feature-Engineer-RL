import requests
import json

# Replace this with your actual Hugging Face Space URL
BASE_URL = "https://sargam-ghagre-feature-engineer-rl.hf.space"

def run_baseline():
    print("--- Starting Baseline Inference ---")
    
    # 1. Reset Environment
    print(f"Step 1: Resetting to 'easy' task...")
    reset_req = requests.post(f"{BASE_URL}/reset?task_id=easy")
    if reset_req.status_code == 200:
        initial_score = reset_req.json()['observation']['current_score']
        print(f"Success! Initial Accuracy: {initial_score:.4f}")
    
    # 2. Take a Step (Action)
    print("\nStep 2: Performing Mean Imputation on 'feature1'...")
    action = {"action_type": "impute_mean", "column": "feature1"}
    step_req = requests.post(f"{BASE_URL}/step", json=action)
    
    if step_req.status_code == 200:
        result = step_req.json()
        new_score = result['observation']['current_score']
        reward = result['reward']
        print(f"Action Complete!")
        print(f"New Accuracy: {new_score:.4f}")
        print(f"Reward Received: {reward:.4f}")
    else:
        print(f"Error: {step_req.text}")

if __name__ == "__main__":
    run_baseline()
