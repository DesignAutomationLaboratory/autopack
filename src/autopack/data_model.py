import pathlib
from typing import Any, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

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
    min_bounding_box: bool = Field(
        default=False,
        description="Whether to use the minimum bounding box for the cost field grid, as calculated by IPS. If False, the default bounding box will be used.",
    )
    custom_bounding_box: Optional[
        tuple[tuple[float, float, float], tuple[float, float, float]]
    ] = Field(
        default=None,
        description="Custom bounding box for the cost field grid. If supplied, must be a pair of points in world coordinates.",
    )
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

    @model_validator(mode="after")
    def check_bounding_box_settings(self):
        if self.min_bounding_box and self.custom_bounding_box:
            raise ValueError(
                "Cannot use both minimum bounding box and custom bounding box"
            )
        return self


class CostField(BaseModel, arbitrary_types_allowed=True):
    name: str
    coordinates: np.ndarray = Field(repr=False)
    costs: np.ndarray = Field(
        repr=False,
        description="Cost for each grid node. Positive infinity implies an infeasible node.",
    )
    ergo: bool = Field(
        default=False,
        description="Whether this is an ergonomy cost field. If so, the name should be the name of the ergo standard.",
    )

    @field_validator("costs")
    @classmethod
    def check_costs(cls, v: np.ndarray):
        assert not np.isnan(v).any(), "Cost field contains NaNs"
        assert not np.isneginf(v).any(), "Cost field contains negative infinity values"

        return v


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
