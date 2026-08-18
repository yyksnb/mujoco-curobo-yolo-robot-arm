from __future__ import annotations

from types import MethodType

from task2.backends.curobo_ik import CuroboIKSolver
from task2.contracts import JointLimits, RigidTransform, RobotModel, RobotState
from task2.ik import IKConstraints, IKResult, IKSolution


def _model() -> RobotModel:
    model = RobotModel(
        "test",
        "base",
        "tcp",
        ("continuous", "bounded"),
        (
            JointLimits(-6.28, 6.28, 1.0, 1.0, 1.0, continuous=True),
            JointLimits(-1.0, 1.0, 1.0, 1.0, 1.0),
        ),
        (0.0, 0.0),
    )
    model.validate()
    return model


def test_curobo_solver_continues_to_deterministic_seed_batch() -> None:
    model = _model()
    solver = object.__new__(CuroboIKSolver)
    solver._model = model
    solver._seed_count = 2
    solver._seed_batch_count = 2
    calls: list[tuple[RobotState, ...]] = []

    def fake_batch(self, target, seed_states):
        calls.append(tuple(seed_states))
        state = RobotState(
            model.joint_names,
            (0.1, 0.0) if len(calls) == 1 else (0.2, 0.0),
        )
        return IKResult(
            (IKSolution(state, 0.0, 0.0),),
            None,
            "fake batch",
        )

    solver._solve_seed_batch = MethodType(fake_batch, solver)
    target = RigidTransform(
        "base",
        "tcp",
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    seed = RobotState(model.joint_names, (0.0, 0.0))
    result = solver.solve(
        (target,),
        (seed,),
        IKConstraints(0.001, 0.01, 1, lambda state: state.positions_rad[0] > 0.15),
    )

    assert result.success
    assert result.solutions[0].state.positions_rad == (0.2, 0.0)
    assert len(calls) == 2
    assert calls[0][0].positions_rad == seed.positions_rad
    assert calls[1][0].positions_rad != seed.positions_rad


def test_curobo_solver_keeps_valid_first_batch_bounded() -> None:
    model = _model()
    solver = object.__new__(CuroboIKSolver)
    solver._model = model
    solver._seed_count = 2
    solver._seed_batch_count = 3
    calls = 0

    def fake_batch(self, target, seed_states):
        nonlocal calls
        calls += 1
        state = RobotState(model.joint_names, (0.2, 0.0))
        return IKResult((IKSolution(state, 0.0, 0.0),), None, "fake batch")

    solver._solve_seed_batch = MethodType(fake_batch, solver)
    target = RigidTransform(
        "base",
        "tcp",
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    seed = RobotState(model.joint_names, (0.0, 0.0))

    result = solver.solve(
        (target,),
        (seed,),
        IKConstraints(0.001, 0.01, 2, lambda state: True),
    )

    assert result.success
    assert calls == 1
