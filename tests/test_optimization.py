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


@pytest.mark.parametrize("use_cuda", [True, False])
def test_minimize_binh_korn_deterministic(use_cuda):
    problem = create_binh_korn_problem(constrained=True)

    init_samples = 8
    batches = 4
    batch_size = 2

    runs = []
    for run in range(2):
        result = minimize(
            problem=problem,
            init_samples=init_samples,
            batches=batches,
            batch_size=batch_size,
            seed=0,
            use_cuda=use_cuda,
        )
        runs.append(result)

    all_xs = np.stack([run.x for run in runs])

    sampled_xs = all_xs[:, :init_samples]
    opt_xs = all_xs[:, init_samples:]

    # Are we sampling the same points?
    assert np.diff(sampled_xs, axis=0).sum() == 0

    # Does the optimizer come up with the same points?
    assert np.diff(opt_xs, axis=0).sum() == 0

    # Do we actually evaluate unique points?
    _, counts = np.unique(all_xs[0], axis=0, return_counts=True)
    assert np.all(counts == 1)

    all_objs = np.stack([run.obj for run in runs])
    all_cons = np.stack([run.con for run in runs])

    # Sanity check, is our function actually deterministic?
    assert np.diff(all_objs, axis=0).sum() == 0
    assert np.diff(all_cons, axis=0).sum() == 0
