import numpy as np
import pytest

from autopack.optimization import OptimizationProblem, OptimizationResult, minimize


def create_binh_korn_problem(constrained):
    def binh_korn(xs, meta=None):
        x_0 = xs[:, 0]
        x_1 = xs[:, 1]
        objs = np.array(
            [
                4 * x_0**2 + 4 * x_1**2,
                (x_0 - 5) ** 2 + (x_1 - 5) ** 2,
            ]
        ).T
        if constrained:
            cons = np.array(
                [
                    (x_0 - 5) ** 2 + x_1**2 - 25,
                    -((x_0 - 8) ** 2) - (x_1 + 3) ** 2 + 7.7,
                ]
            ).T
        else:
            cons = np.empty((xs.shape[0], 0), dtype=float)

        return xs, objs, cons

    return OptimizationProblem(
        func=binh_korn,
        bounds=np.array([[0.0, 5.0], [0.0, 3.0]]),
        num_objectives=2,
        num_constraints=2 if constrained else 0,
        # ref_point=np.array([140, 50]),
    )


@pytest.mark.parametrize("constrained", [True, False])
def test_minimize_binh_korn_smoke(constrained):
    problem = create_binh_korn_problem(constrained)

    # Smoke test, just make sure it runs at all
    result = minimize(
        problem=problem,
        batches=2,
        batch_size=2,
    )

    assert isinstance(result, OptimizationResult)
    assert result.x.shape[0] >= 4
    assert result.obj.shape[0] == result.x.shape[0]
    assert result.con.shape[0] == result.x.shape[0]
    assert result.x.shape[1] == 2
    assert result.obj.shape[1] == 2
    assert result.con.shape[1] == (2 if constrained else 0)


@pytest.mark.parametrize("num_weights", [2, 3, 4])
@pytest.mark.parametrize("cost_multiplier", [10])
@pytest.mark.parametrize("clip_multiplier", [1])
def test_faux_autopack_smoke(num_weights, cost_multiplier, clip_multiplier):
    datasets = []

    problem = create_faux_autopack_analysis_problem(
        num_weights=num_weights,
        cost_multiplier=cost_multiplier,
        clip_multiplier=clip_multiplier,
        datasets=datasets,
    )

    result = minimize(
        problem=problem,
        batches=2,
        batch_size=2,
    )

    full_dataset = xr.concat(datasets, dim="case")

    # Check that the hypervolume is increasing, if ever so slightly
    assert False
