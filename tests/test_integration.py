import numpy as np
import pytest
import xarray as xr
from facit.testing import assert_ds_equal

from autopack.data_model import Cable, CostField, Geometry, HarnessSetup, ProblemSetup
from autopack.default_commands import create_default_prob_setup
from autopack.harness_optimization import (
    global_optimize_harness,
)
from autopack.io import load_dataset, save_dataset
from autopack.ips_communication.ips_commands import create_costfield, load_scene


def test_harness_optimization_setup(simple_plate_harness_setup):
    cost_field = CostField(
        name="test",
        coordinates=np.ones((1, 1, 1, 3), dtype=float),
        costs=np.ones((1, 1, 1), dtype=float),
    )

    opt_setup = ProblemSetup(
        harness_setup=simple_plate_harness_setup, cost_fields=[cost_field]
    )


def test_global_optimization_smoke(simple_plate_harness_setup, ips_instance, tmpdir):
    problem_setup = create_default_prob_setup(
        ips_instance=ips_instance,
        harness_setup=simple_plate_harness_setup,
        create_imma=True,
    )

    dataset = global_optimize_harness(
        ips_instance=ips_instance,
        problem_setup=problem_setup,
        init_samples=2,
        batches=2,
        batch_size=2,
    )

    assert isinstance(dataset, xr.Dataset)
    assert dataset.attrs["problem_setup"] == problem_setup
    assert dataset.attrs["ips_version"] == ips_instance.version
    assert len(dataset.solution) >= 6

    ds_path = tmpdir / "test_dataset"
    save_dataset(dataset, ds_path)
    loaded_dataset = load_dataset(ds_path)

    # FIXME: trips on comparing arrays inside Harness objects
    # assert_ds_equal(dataset, loaded_dataset)


def test_create_problem_setup_without_imma(ips_instance, simple_plate_harness_setup):
    problem_setup = create_default_prob_setup(
        ips_instance=ips_instance,
        harness_setup=simple_plate_harness_setup,
        create_imma=False,
    )

    assert isinstance(problem_setup, ProblemSetup)
    assert len(problem_setup.cost_fields) == 1


def test_create_problem_setup_with_imma(ips_instance, simple_plate_harness_setup):
    problem_setup = create_default_prob_setup(
        ips_instance=ips_instance,
        harness_setup=simple_plate_harness_setup,
        create_imma=True,
    )

    assert isinstance(problem_setup, ProblemSetup)
    assert len(problem_setup.cost_fields) == 3
