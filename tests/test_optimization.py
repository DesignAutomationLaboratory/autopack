import numpy as np
import pytest

from autopack.optimization import OptimizationProblem, OptimizationResult, minimize


def create_binh_korn_problem(constrained):
    def binh_korn_obj(x):
        return np.array(
            [
                4 * x[0] ** 2 + 4 * x[1] ** 2,
                (x[0] - 5) ** 2 + (x[1] - 5) ** 2,
            ]
        )

    def binh_korn_con(x):
        return np.array(
            [
                (x[0] - 5) ** 2 + x[1] ** 2 - 25,
                -((x[0] - 8) ** 2) - (x[1] + 3) ** 2 + 7.7,
            ]
        )

    return OptimizationProblem(
        obj_func=binh_korn_obj,
        con_func=binh_korn_con if constrained else None,
        bounds=np.array([[0, 5], [0, 3]]),
        num_objectives=2,
        num_constraints=2 if constrained else 0,
        ref_point=np.array([140, 50]),
    )


@pytest.mark.parametrize("constrained", [True, False])
def test_minimize_smoke(constrained):
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
