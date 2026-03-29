import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from models import Action, Observation, StepOutput

class FeatureEngineeringEnv:
    def __init__(self, df, target):
        self.original_df = df.copy()
        self.target = target
        self.max_steps = 10
        self.reset()

    def reset(self):
        self.df = self.original_df.copy()
        self.steps = 0
        self.last_score = self._get_score()
        return self.get_state()

    def _get_score(self):
        try:
            if self.df.empty or len(self.df.columns) <= 1:
                return 0.0
            
            temp_df = self.df.copy()
            X = temp_df.drop(columns=[self.target])
            y = temp_df[self.target]

            X = X.fillna(0)
            X = pd.get_dummies(X)

            if X.empty or X.shape[1] == 0:
                return 0.0

            model = RandomForestClassifier(n_estimators=10, random_state=42)
            scores = cross_val_score(model, X, y, cv=3)
            return float(np.mean(scores))
        except Exception:
            return 0.0

    def step(self, action: Action):
        self.steps += 1
        reward = -0.01 

        if action.column and action.column not in self.df.columns:
            return StepOutput(self.get_state(), -0.05, False, {"error": "Column not found"})

        if action.action_type == "impute_mean" and action.column:
            if pd.api.types.is_numeric_dtype(self.df[action.column]):
                self.df[action.column] = self.df[action.column].fillna(self.df[action.column].mean())
            else:
                reward -= 0.05 
        elif action.action_type == "drop_column" and action.column:
            if action.column != self.target and len(self.df.columns) > 1:
                self.df = self.df.drop(columns=[action.column])
            else:
                reward -= 0.05

        new_score = self._get_score()
        reward += (new_score - self.last_score)
        self.last_score = new_score

        done = self.steps >= self.max_steps or action.action_type == "submit"
        
        return StepOutput(
            observation=self.get_state(),
            reward=float(reward),
            done=done,
            info={"current_model_score": new_score},
        )

    def get_state(self):
        metadata = {
            col: {
                "dtype": str(self.df[col].dtype),
                "missing": int(self.df[col].isnull().sum()),
                "missing_pct": float(self.df[col].isnull().mean())
            }
            for col in self.df.columns if col != self.target
        }
        return Observation(
            column_metadata=metadata,
            current_score=float(self.last_score),
            steps_left=self.max_steps - self.steps,
            task_goal="Optimize Titanic Survival Prediction Accuracy",
        )
