import sys

def run_inference():
    task_name = "Feature-Engineer-RL"
    
    # 1. [START] Block
    print(f"[START] task={task_name}", flush=True)
    
    # Your environment setup (Gymnasium/OpenEnv)
    # state = env.reset()
    
    total_reward = 0
    for step in range(1, 11):  # Example: 10 steps
        # logic: action = agent.predict(state)
        # logic: state, reward, done, info = env.step(action)
        
        current_reward = 0.5 # Replace with your actual step reward
        total_reward += current_reward
        
        # 2. [STEP] Block - Must be printed every step
        print(f"[STEP] step={step} reward={current_reward}", flush=True)
        
        # if done: break

    # 3. [END] Block - Final summary
    final_score = total_reward # Or your specific metric
    print(f"[END] task={task_name} score={final_score} steps={step}", flush=True)

if __name__ == "__main__":
    run_inference()
