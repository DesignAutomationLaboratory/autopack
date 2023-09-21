from typing import Literal, List, Optional, Any
from pydantic import BaseModel
import json
import numpy as np

class Cable(BaseModel):
    start_node: str
    end_node: str
    cable_type: str

class Geometry(BaseModel):
    name: str
    clearance: float
    preference: Literal['Near', 'Avoid', 'Neutral']
    clipable: bool
    assembly: bool

class HarnessSetup(BaseModel):
    geometries: List[Geometry]
    cables:List[Cable]

class CostField(BaseModel, arbitrary_types_allowed=True):
    name: str
    coordinates: np.ndarray 
    costs: np.ndarray
    def normalized_costs(self):
        return None

class ProblemSetup(BaseModel):
    harness_setup: HarnessSetup
    cost_fields: List[CostField]

class HarnessSegment(BaseModel):
    cables: list[int]
    points: list[tuple[int, int, int]]

class Harness(BaseModel):
    harness_segments: list[HarnessSegment]
    numb_of_clips: int = 0

class Result(BaseModel):
    problem_setup: ProblemSetup
    dataset: Any
    ips_path: str
    scene_path: str