import pathlib
from typing import Any, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from .ips_communication.ips_class import IPSInstance  # noqa


class Cable(BaseModel):
    start_node: str
    end_node: str
    cable_type: str


class Geometry(BaseModel):
    name: str | Literal["Static Geometry"] = Field(
        description="Name of a rigid body or 'Static Geometry' for the static geometry root"
    )
    clearance: float = Field(description="Required clearance in millimeters")
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
    grid_resolution: float = Field(
        default=0.02,
        description="Grid resolution in meters.",
    )
    allow_infeasible_topology: bool = Field(
        default=False,
        description="Whether to allow infeasible harness topologies.",
    )

    @classmethod
    def from_json_file(cls, json_path: pathlib.Path):
        return cls.model_validate_json(json_path.read_text())


class CostField(BaseModel, arbitrary_types_allowed=True):
    name: str
    coordinates: np.ndarray = Field(repr=False)
    costs: np.ndarray = Field(
        repr=False,
        description="Cost for each grid node. Positive infinity implies an infeasible node.",
    )

    _interpolator: Optional[Any] = Field(
        exclude=True,
        description="An interpolator object that interpolates the cost field",
    )

    @property
    def interpolator(self):
        if self._interpolator is None:
            raise ValueError("Cost field has no interpolator")
        return self._interpolator

    def normalized_costs(self):
        feasible_mask = np.invert(np.isposinf(self.costs))
        max_value = np.amax(self.costs[feasible_mask])
        assert max_value != 0, "Max cost is 0, can't normalize"
        normalized_arr = self.costs / max_value
        return normalized_arr


class ProblemSetup(BaseModel):
    harness_setup: HarnessSetup
    cost_fields: List[CostField]

    @property
    def ref_cost_field(self):
        """
        Reference cost field
        """
        return self.cost_fields[0]


class HarnessSegment(BaseModel, arbitrary_types_allowed=True):
    cables: list[int] = Field(
        description="Indexes of cables that are included in the segment"
    )
    radius: float = Field(description="Total radius of the segment")
    discrete_nodes: np.ndarray = Field(
        description="Grid node indexes that the discrete solution visits", repr=False
    )
    discrete_coords: np.ndarray = Field(
        description="Coordinates that the discrete solution visits", repr=False
    )
    presmooth_coords: np.ndarray = Field(
        description="Coordinates that the presmoothed solution visits", repr=False
    )
    smooth_coords: np.ndarray = Field(
        description="Coordinates that the smoothed solution visits, if available",
        repr=False,
    )
    clip_coords: np.ndarray = Field(
        description="Positions of clips from smoothed solution, if available",
        repr=False,
    )

    @property
    def points(self):
        # For backwards compatibility
        return self.discrete_nodes


class Harness(BaseModel):
    name: str
    topology_feasible: bool = Field(
        description="Whether the harness topology is feasible"
    )
    harness_segments: list[HarnessSegment]
    num_branch_points: int = Field(description="Number of branch points")
    cable_segment_order: list[list[int]] = Field(
        description="Order in which segments are visited, per cable"
    )
    bundling_weight: float = Field(description="The bundling weight as returned by IPS")
    bundling_objective: float = Field(
        description="The bundling objective as returned by IPS"
    )
    length_objective: float = Field(
        description="The length objective as returned by IPS"
    )
    length_total: float = Field(description="The total length of the harness in meters")
    length_in_collision: float = Field(
        description="The length of the harness in collision in meters"
    )


class Result(BaseModel):
    problem_setup: ProblemSetup
    dataset: Any


class GlobalOptimizationSetup(BaseModel):
    problem_setup: ProblemSetup
    ips_instance: Any
