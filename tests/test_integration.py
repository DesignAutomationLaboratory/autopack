import numpy as np
import pytest
import xarray as xr
from facit.testing import assert_ds_equal

from autopack.data_model import Cable, CostField, Geometry, HarnessSetup, ProblemSetup
from autopack.default_commands import create_default_prob_setup
from autopack.harness_optimization import route_evaluate_harness
from autopack.io import load_dataset, save_dataset
from autopack.ips_communication.ips_commands import create_costfield, load_scene
from autopack.optimization import global_optimize_harness


@pytest.fixture
def simple_plate_harness_setup(test_scenes_path):
    scene_path = str((test_scenes_path / "simple_plate.ips").resolve())

    cable1 = Cable(
        start_node="Cable1_start",
        end_node="Cable1_end",
        cable_type="ID_Rubber_template",
    )
    cable2 = Cable(
        start_node="Cable2_start",
        end_node="Cable2_end",
        cable_type="ID_Rubber_template",
    )
    cable3 = Cable(
        start_node="Cable3_start",
        end_node="Cable3_end",
        cable_type="ID_Rubber_template",
    )
    part1 = Geometry(
        name="part1", clearance=5.0, preference="Near", clipable=True, assembly=True
    )
    part2 = Geometry(
        name="part2", clearance=5.0, preference="Near", clipable=True, assembly=True
    )
    part3 = Geometry(
        name="part3", clearance=5.0, preference="Near", clipable=True, assembly=True
    )
    return HarnessSetup(
        scene_path=scene_path,
        geometries=[part1, part2, part3],
        cables=[cable1, cable2, cable3],
    )


def test_harness_optimization_setup(simple_plate_harness_setup):
    cost_field = CostField(
        name="test",
        coordinates=np.ones((1, 1, 1, 3), dtype=float),
        costs=np.ones((1, 1, 1), dtype=float),
    )

    opt_setup = ProblemSetup(
        harness_setup=simple_plate_harness_setup, cost_fields=[cost_field]
    )


def test_route_evaluate_harness(simple_plate_harness_setup, ips_instance):
    load_scene(ips_instance, simple_plate_harness_setup.scene_path)

    cost_field_ips, cost_field_length = create_costfield(
        ips_instance, simple_plate_harness_setup
    )
    opt_setup = ProblemSetup(
        harness_setup=simple_plate_harness_setup,
        cost_fields=[cost_field_ips, cost_field_length],
    )

    bundling_costs, total_costs, numb_of_clips = route_evaluate_harness(
        ips_instance, opt_setup, [0.5, 0.5], 0.5, harness_id="test"
    )

    assert bundling_costs.shape == (1, 2)
    assert total_costs.shape == (1, 2)
    assert numb_of_clips.shape == (1,)
    assert np.all(bundling_costs >= 0)
    assert np.all(total_costs >= 0)
    assert np.all(numb_of_clips >= 0)


def test_global_optimization_smoke(simple_plate_harness_setup, ips_instance, tmpdir):
    problem_setup = create_default_prob_setup(
        ips_instance=ips_instance,
        harness_setup=simple_plate_harness_setup,
        create_imma=False,
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

    ds_path = tmpdir / "test.zarr"
    save_dataset(dataset, ds_path)
    loaded_dataset = load_dataset(ds_path)

    assert_ds_equal(dataset, loaded_dataset)


def test_create_problem_setup_without_imma(ips_instance, simple_plate_harness_setup):
    problem_setup = create_default_prob_setup(
        ips_instance=ips_instance,
        harness_setup=simple_plate_harness_setup,
        create_imma=False,
    )

    assert isinstance(problem_setup, ProblemSetup)
    assert len(problem_setup.cost_fields) == 2


def test_create_problem_setup_with_imma(ips_instance, simple_plate_harness_setup):
    problem_setup = create_default_prob_setup(
        ips_instance=ips_instance,
        harness_setup=simple_plate_harness_setup,
        create_imma=True,
    )

    assert isinstance(problem_setup, ProblemSetup)
    assert len(problem_setup.cost_fields) == 4
