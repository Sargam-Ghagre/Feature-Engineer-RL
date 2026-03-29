# Feature Engineering RL Agent

An autonomous Reinforcement Learning (RL) agent designed to "play" data preprocessing like a game. The agent explores various feature engineering strategies—such as mean imputation and column dropping—to maximize the predictive accuracy of a machine learning model.

## Live Demo
The project is deployed and interactive on Hugging Face Spaces:
**[View Hugging Face Space](https://huggingface.co/spaces/Sargam-Ghagre/Feature-Engineer-RL)**

---

## Project Architecture

This project implements a full MLOps pipeline:
1. **RL Environment**: A custom OpenAI Gym-style environment (`env.py`) that transforms data and calculates rewards.
2. **The Agent**: A Proximal Policy Optimization (PPO) model trained using `stable-baselines3`.
3. **Reward Signal**: The agent receives a reward based on the change in a Random Forest Classifier's cross-validation score after each action.
4. **API Layer**: A **FastAPI** backend that exposes the environment to external users/tools.
5. **Deployment**: Fully containerized using **Docker**.

---

## Tech Stack
* **Language**: Python 3.9+
* **RL Framework**: Stable-Baselines3 (PPO)
* **ML Libraries**: Scikit-Learn, Pandas, Numpy
* **API Framework**: FastAPI & Uvicorn
* **Deployment**: Docker & Hugging Face Spaces

---

## Quick Start (API Usage)

Once the Space is running, you can interact with the agent via the following endpoints:

### 1. Reset the Environment
Initializes a new task (e.g., Titanic dataset).
- **Endpoint**: `POST /reset?task_id=easy`

### 2. Take an Action
Tell the agent to perform a specific engineering task.
- **Endpoint**: `POST /step`
- **Payload**:
```json
{
  "action_type": "impute_mean",
  "column": "Age"
}
