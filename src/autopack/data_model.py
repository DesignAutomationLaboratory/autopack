import json
from typing import Any, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from .ips_communication.ips_class import IPSInstance  # noqa


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
    scene_path: str  # Absolute path to scene
    geometries: List[Geometry]
    cables: List[Cable]
    clip_clip_dist: tuple[float, float] = (
        0.0375,
        0.15,
    )  # min/max distance between clips
    branch_clip_dist: tuple[float, float] = (
        0.01875,
        0.075,
    )  # min/max distance between branch and clip


class CostField(BaseModel, arbitrary_types_allowed=True):
    name: str
    coordinates: np.ndarray = Field(repr=False)
    costs: np.ndarray = Field(repr=False)

    def normalized_costs(self):
        mask = self.costs < 999999
        max_value = np.amax(self.costs[mask])
        assert max_value != 0, "Max cost is 0, can't normalize"
        normalized_arr = self.costs / max_value
        return normalized_arr


class ProblemSetup(BaseModel):
    harness_setup: HarnessSetup
    cost_fields: List[CostField]


class HarnessSegment(BaseModel):
    cables: list[int] = Field(
        description="Indexes of cables that are included in the segment"
    )
    radius: float = Field(description="Total radius of the segment")
    points: list[tuple[int, int, int]] = Field(
        description="Grid node indexes that the discrete solution visits", repr=False
    )
    presmooth_coords: list[tuple[float, float, float]] = Field(
        description="Coordinates that the presmoothed solution visits", repr=False
    )
    smooth_coords: Optional[list[tuple[float, float, float]]] = Field(
        description="Coordinates that the smoothed solution visits, if available",
        repr=False,
    )
    clip_positions: Optional[list[tuple[float, float, float]]] = Field(
        description="Positions of clips from smoothed solution, if available",
        repr=False,
    )


class Harness(BaseModel):
    name: str
    harness_segments: list[HarnessSegment]
    numb_of_clips: int = Field(description="Estimated number of clips")
    num_branch_points: int = Field(description="Number of branch points")
    bundling_factor: float = Field(description="The bundling factor as returned by IPS")
    bundling_objective: float = Field(
        description="The bundling objective as returned by IPS"
    )
    length_objective: float = Field(
        description="The length objective as returned by IPS"
    )


class Result(BaseModel):
    problem_setup: ProblemSetup
    dataset: Any


class GlobalOptimizationSetup(BaseModel):
    problem_setup: ProblemSetup
    ips_instance: Any
