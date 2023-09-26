import json
from typing import Any, List, Literal, Optional

import numpy as np
from pydantic import BaseModel


class Cable(BaseModel):
    start_node: str
    end_node: str
    cable_type: str


class Geometry(BaseModel):
    name: str
    clearance: float
    preference: Literal["Near", "Avoid", "Neutral"]
    clipable: bool
    assembly: bool


class HarnessSetup(BaseModel, arbitrary_types_allowed=True):
    geometries: List[Geometry]
    cables: List[Cable]
    clip_clip_dist: tuple[float, float] = (0.15, 0.45)  # min/max distance between clips
    branch_clip_dist: tuple[float, float] = (
        0.05,
        2,
    )  # min/max distance between branch and clip


class CostField(BaseModel, arbitrary_types_allowed=True):
    name: str
    coordinates: np.ndarray
    costs: np.ndarray

    def normalized_costs(self):
        mask = self.costs < 999999
        max_value = np.amax(self.costs[mask])
        normalized_arr = self.costs / max_value
        return normalized_arr


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
