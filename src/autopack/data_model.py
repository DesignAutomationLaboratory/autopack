import os
import pathlib
from typing import Any, List, Literal, Optional

import numpy as np
import scipy as sp
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


class HarnessSetup(BaseModel):
    scene_path: pathlib.Path = Field(
        description="Path to the IPS scene file. Can be absolute or relative to the harness setup JSON file.",
    )
    geometries: List[Geometry]
    cables: List[Cable]
    clip_clip_dist: tuple[float, float] = Field(
        default=(
            0.0375,
            0.15,
        ),
        description="Min/max distance between clips",
    )
    branch_clip_dist: tuple[float, float] = Field(
        default=(
            0.01875,
            0.075,
        ),
        description="Min/max distance between branch point and clip",
    )
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
    grid_size: int = Field(
        default=50,
        # Minimum required by IPS
        ge=5,
        description="Number of grid nodes in the longest side of the minimum bounding box.",
    )
    allow_infeasible_topology: bool = Field(
        default=False,
        description="Whether to allow infeasible harness topologies.",
    )

    @classmethod
    def from_json_file(cls, json_path: pathlib.Path):
        obj = cls.model_validate_json(json_path.read_text())
        if not os.path.isabs(obj.scene_path):
            obj.scene_path = json_path.parent / obj.scene_path
        return obj

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

    @property
    def interpolator(self) -> sp.interpolate.NearestNDInterpolator:
        """
        Interpolator for cost field
        """
        if hasattr(self, "_interpolator"):
            return self._interpolator

        flat_coords = self.coordinates.reshape(-1, 3)
        flat_costs = self.costs.reshape(-1)
        feasible_mask = np.isfinite(flat_costs)
        feasible_costs = flat_costs[feasible_mask]
        feasible_coords = flat_coords[feasible_mask]

        interpolator = sp.interpolate.NearestNDInterpolator(
            feasible_coords, feasible_costs
        )
        self._interpolator = interpolator

        return interpolator

    def interpolate(self, coords: np.ndarray):
        """
        Interpolate cost field at given coordinates (shape (n_points, 3))
        """
        return self.interpolator(coords)

    @property
    def dimensions(self):
        """
        Cost field dimensions (x, y, z) in meters.
        """
        coords = self.coordinates.reshape(-1, 3)
        return np.ptp(coords, axis=0)

    @property
    def size(self):
        """
        Cost field grid size (x, y, z) in number of nodes.
        """
        return np.array(self.costs.shape)

    @property
    def resolution(self):
        """
        Cost field grid resolution in meters per grid cell.
        """
        return np.mean(self.dimensions / self.size)


class ErgoSettings(BaseModel):
    grip_tolerance: float = Field(
        default=0.01,
        gt=0.0,
        description="Grip tolerance, in meters. Max distance between grip and target position.",
    )
    sample_ratio: float = Field(
        default=1 / 10,
        gt=0,
        le=1,
        description="Ratio for sampling a number of grid points to evaluate for ergo.",
    )
    min_samples: int = Field(
        # More reasonable default
        default=24,
        # Hard requirement for the RBF interpolator
        ge=4,
        description="Minimum number of samples to evaluate for ergo. Used mainly for testing. Change the sample_ratio instead.",
    )
    use_rbpp: bool = Field(
        default=True,
        title="Use Rigid Body Path Planning",
        description="Whether IPS's Rigid Body Path Planning should be used when evaluating manikins. Takes more time but should be more robust in finding feasible grips.",
    )


class ProblemSetup(BaseModel):
    harness_setup: HarnessSetup
    ergo_settings: Optional[ErgoSettings] = Field(default=None)
    cost_fields: List[CostField]

    @property
    def ref_cost_field(self):
        """
        Reference cost field
        """
        return self.cost_fields[0]


class StudySettings(BaseModel):
    seed_with_ips_samples: bool = Field(
        default=True,
        description="Whether to seed the study with IPS-style samples",
    )
    doe_samples: int = Field(
        default=16,
        ge=2,
    )
    opt_batches: int = Field(
        default=8,
        ge=0,
    )
    opt_batch_size: int = Field(
        default=8,
        ge=1,
    )
    random_seed: Optional[int] = Field(default=0)
    silence_warnings: bool = Field(default=True)


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

    @property
    def all_clip_coords(self):
        return np.concatenate([seg.clip_coords for seg in self.harness_segments])
