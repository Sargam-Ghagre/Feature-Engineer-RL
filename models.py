from pydantic import BaseModel
from typing import Dict, Optional, Any

class Action(BaseModel):
    action_type: str  # Options: "impute_mean", "drop_column", "submit"
    column: Optional[str] = None

class Observation(BaseModel):
    column_metadata: Dict[str, Any]
    current_score: float
    steps_left: int
    task_goal: str

class StepOutput(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]
