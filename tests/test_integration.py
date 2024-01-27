import numpy as np
import pytest
import xarray as xr
from facit.testing import assert_ds_equal

from autopack import __version__
from autopack.data_model import (
    CostField,
    ErgoSettings,
    ProblemSetup,
    StudySettings,
)
from autopack.io import load_dataset, save_dataset
from autopack.workflows import build_problem, build_problem_and_run_study


def test_harness_optimization_setup(simple_plate_harness_setup):
    cost_field = CostField(
        name="test",
        coordinates=np.ones((1, 1, 1, 3), dtype=float),
        costs=np.ones((1, 1, 1), dtype=float),
    )

    opt_setup = ProblemSetup(
        harness_setup=simple_plate_harness_setup, cost_fields=[cost_field]
    )


@pytest.mark.parametrize("run_ergo", [False, True])
def test_global_optimization_smoke(simple_plate_harness_setup, ips, tmpdir, run_ergo):
    if run_ergo:
        ergo_settings = ErgoSettings(
            sample_ratio=0.01,
            min_samples=4,
        )
    else:
        ergo_settings = None

    study_settings = StudySettings(
        doe_samples=4,
        opt_batches=2,
        opt_batch_size=2,
        # Errors should not be swallowed
        return_partial_results=False,
    )

    dataset = build_problem_and_run_study(
        ips=ips,
        harness_setup=simple_plate_harness_setup,
        ergo_settings=ergo_settings,
        study_settings=study_settings,
    )

    assert isinstance(dataset, xr.Dataset)
    assert dataset.attrs["autopack_version"] == __version__
    assert dataset.attrs["ips_version"] == ips.version
    assert dataset.attrs["problem_setup"].harness_setup == simple_plate_harness_setup
    assert dataset.attrs["problem_setup"].ergo_settings == ergo_settings
    assert dataset.attrs["study_settings"] == study_settings
    assert len(dataset.solution) >= 8

    ds_path = tmpdir / "test_dataset"
    save_dataset(dataset, ds_path)
    loaded_dataset = load_dataset(ds_path)

    # FIXME: trips on comparing arrays inside Harness objects
    # assert_ds_equal(dataset, loaded_dataset)


@pytest.mark.parametrize("run_ergo", [False, True])
def test_build_problem(ips, simple_plate_harness_setup, run_ergo):
    if run_ergo:
        ergo_settings = ErgoSettings(
            sample_ratio=0.01,
            min_samples=4,
        )
    else:
        ergo_settings = None
    problem_setup = build_problem(
        ips=ips,
        harness_setup=simple_plate_harness_setup,
        ergo_settings=ergo_settings,
    )

    assert isinstance(problem_setup, ProblemSetup)
    assert len(problem_setup.cost_fields) == 3 if run_ergo else 1
