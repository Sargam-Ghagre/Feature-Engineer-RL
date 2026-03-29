import pandas as pd
import numpy as np
from env import FeatureEngineeringEnv

def load_task(task_level: str):
    # Seed ensures the judge sees the same 'random' data every time
    np.random.seed(42)
    
    if task_level == "easy":
        # Simple task: Just fix missing values in one column
        df = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "target": np.random.randint(0, 2, 100),
        })
        df.loc[::5, "feature1"] = np.nan # 20% missing
        
    elif task_level == "medium":
        # Medium task: High cardinality categories + some missingness
        df = pd.DataFrame({
            "category": np.random.choice([f"cat{i}" for i in range(20)], 200),
            "num": np.random.randn(200),
            "target": np.random.randint(0, 2, 200),
        })
        df.loc[::10, "num"] = np.nan
        
    elif task_level == "hard":
        # Hard task: Redundant columns (Multicollinearity) + Missing Data
        x = np.random.randn(500)
        df = pd.DataFrame({
            "x1": x,
            "x2": x * 0.99 + np.random.normal(0, 0.01, 500), # Redundant column
            "noisy_feat": np.random.randn(500), # Useless noise
            "target": (x > 0).astype(int),
        })
        # Add a lot of missing values to force the agent to work
        df.loc[::3, "x1"] = np.nan 
        df.loc[::4, "noisy_feat"] = np.nan
    else:
        raise ValueError(f"Invalid task level: {task_level}")
        
    # Return the environment with the specific dataset and the name of the target
    return FeatureEngineeringEnv(df, "target")
