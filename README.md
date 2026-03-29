title: Feature-Engineer-RL
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
Feature-Engineer-RL
Autonomous Reinforcement Learning for Data Optimization
This repository contains an autonomous Reinforcement Learning (RL) agent specifically designed to solve the Titanic Survival Prediction task through automated feature engineering.

Developed for the OpenEnv 2026 Hackathon, this agent treats data preprocessing as a sequential decision-making game, optimizing for model accuracy within a strictly defined 10-step limit.

The Agent: PPO (Proximal Policy Optimization)
The "brain" of this project is a Stable-Baselines3 PPO agent.

Goal: Maximize the Random Forest Cross-Validation score.
Reward Function: $Reward = \Delta Score - Step_Penalty$.
Observation Space: Rich column-level metadata including dtypes, missing value counts, and current model performance.
Action Space: Discrete choices including Mean Imputation, Column Dropping, and Final Submission.
Technical Architecture
This Space is built using a modern, containerized stack:

Backend: FastAPI (Python 3.11)
RL Framework: Gymnasium & Stable-Baselines3
Data Science: Pandas, NumPy, Scikit-Learn
Containerization: Docker (optimized for Hugging Face Spaces)
API Endpoints (OpenEnv Compliant)
The agent communicates via a REST API:

POST /reset: Initializes the Titanic environment (Easy, Medium, or Hard).
POST /step: Executes an RL-chosen action on the dataset.
GET /state: Retrieves current environment metadata and step counts.
GET /: Health check and status.
How to Run Locally
If you want to test the inference script manually:

Clone the Space.
Ensure you have feature_engineer_agent.zip in the root.
Run:
pip install -r requirements.txt
python inference.py
