"""Pose error and damped least squares solver."""

import mujoco
import numpy as np
import pytest
from numpy.typing import NDArray

from diffik import model as model_mod
from diffik.config import DiffIKConfig
from diffik.model import RobotHandles
from diffik.solver import DiffIKSolver

# Offset from the home pose, about 4.7 cm. Small enough to stay well inside the
# workspace, large enough that a broken solver cannot pass by doing nothing.
NEAR_TARGET_OFFSET = np.array([0.03, 0.03, 0.02])

# Far outside the Panda's roughly 0.85 m reach.
UNREACHABLE_TARGET = np.array([3.0, 0.0, 0.5])


def _place_target(handles: RobotHandles, offset: NDArray[np.float64]) -> None:
    """Start from a zero-error state, then displace the target."""
    model_mod.reset_to_home(handles)
    model_mod.sync_target_to_site(handles)
    handles.data.mocap_pos[handles.mocap_id] += offset


def _error_norms(solver: DiffIKSolver) -> tuple[float, float]:
    error = solver.pose_error.compute()
    return float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))


def _iterate_kinematically(solver: DiffIKSolver, iterations: int) -> None:
    """Run the IK iteration without physics, by writing the integrated
    configuration straight back into qpos.

    The library never does this: DiffIKSolver.step writes to data.ctrl and lets
    the position actuators track the setpoint. Here the point is to measure the
    solver on its own, with no actuator dynamics between the command and the
    configuration.
    """
    handles = solver.handles
    model, data = handles.model, handles.data
    q = np.zeros(model.nq)
    dq_full = np.zeros(model.nv)

    for _ in range(iterations):
        dq = solver.solve(solver.pose_error.compute())
        dq_full[handles.dof_ids] = dq
        np.copyto(q, data.qpos)
        mujoco.mj_integratePos(model, q, dq_full, solver.config.integration_dt)
        data.qpos[:] = q
        mujoco.mj_forward(model, data)


def _simulate(solver: DiffIKSolver, steps: int) -> None:
    for _ in range(steps):
        solver.step()
        mujoco.mj_step(solver.handles.model, solver.handles.data)


# --------------------------------------------------------------------------
# PoseError
# --------------------------------------------------------------------------


def test_error_is_zero_when_target_sits_on_the_site(handles: RobotHandles) -> None:
    solver = DiffIKSolver(handles, DiffIKConfig())
    model_mod.sync_target_to_site(handles)
    error = solver.pose_error.compute()
    assert np.linalg.norm(error) < 1e-12


def test_error_has_the_right_shape_and_sign(handles: RobotHandles) -> None:
    solver = DiffIKSolver(handles, DiffIKConfig())
    _place_target(handles, NEAR_TARGET_OFFSET)
    error = solver.pose_error.compute()
    assert error.shape == (6,)
    # Position error points from the site toward the target.
    assert np.allclose(error[:3], NEAR_TARGET_OFFSET)


def test_error_buffer_is_reused(handles: RobotHandles) -> None:
    """compute() must not allocate: the control loop calls it every step."""
    solver = DiffIKSolver(handles, DiffIKConfig())
    first = solver.pose_error.compute()
    second = solver.pose_error.compute()
    assert first is second is solver.pose_error.value


def test_orientation_error_is_a_rotation_vector(handles: RobotHandles) -> None:
    """Rotating the target by a known angle about a known axis must show up as
    that same axis-angle vector, not as a quaternion difference."""
    solver = DiffIKSolver(handles, DiffIKConfig())
    model_mod.reset_to_home(handles)
    model_mod.sync_target_to_site(handles)

    angle = 0.3
    axis = np.array([0.0, 0.0, 1.0])
    rotation = np.zeros(4)
    mujoco.mju_axisAngle2Quat(rotation, axis, angle)
    turned = np.zeros(4)
    mujoco.mju_mulQuat(turned, rotation, handles.data.mocap_quat[handles.mocap_id])
    handles.data.mocap_quat[handles.mocap_id] = turned

    error = solver.pose_error.compute()
    assert np.allclose(error[3:], angle * axis, atol=1e-9)


# --------------------------------------------------------------------------
# Solver output
# --------------------------------------------------------------------------


def test_solve_returns_seven_joint_velocities(handles: RobotHandles) -> None:
    solver = DiffIKSolver(handles, DiffIKConfig())
    _place_target(handles, NEAR_TARGET_OFFSET)
    dq = solver.solve(solver.pose_error.compute())
    assert dq.shape == (7,)
    assert np.all(np.isfinite(dq))


def test_dls_approaches_the_pseudo_inverse_away_from_singularities(
    handles: RobotHandles,
) -> None:
    """With small damping and a well-conditioned Jacobian the damped solution
    should be close to the minimum-norm one. pinv appears here only as a
    reference; the solver itself must never use it."""
    solver = DiffIKSolver(handles, DiffIKConfig(damping=1e-6))
    _place_target(handles, NEAR_TARGET_OFFSET)
    error = solver.pose_error.compute()

    dq = solver.solve(error).copy()
    reference = np.linalg.pinv(solver.jacobian()) @ error
    assert np.allclose(dq, reference, atol=1e-6)


def test_damping_shrinks_the_joint_velocity(handles: RobotHandles) -> None:
    """More damping must produce a smaller step for the same error. This is the
    trade the method is built on."""
    _place_target(handles, NEAR_TARGET_OFFSET)
    peaks = []
    for damping in (1e-4, 1e-2, 1e-1):
        solver = DiffIKSolver(handles, DiffIKConfig(damping=damping))
        peaks.append(float(np.abs(solver.solve(solver.pose_error.compute())).max()))
    assert peaks[0] > peaks[1] > peaks[2]


def test_max_angvel_clamps_without_turning_the_motion(
    handles: RobotHandles,
) -> None:
    """The clamp scales the whole vector, so the direction survives. Clamping
    entry by entry would steer the end-effector somewhere else."""
    _place_target(handles, UNREACHABLE_TARGET - handles.data.mocap_pos[handles.mocap_id])

    unclamped = DiffIKSolver(handles, DiffIKConfig())
    raw = unclamped.solve(unclamped.pose_error.compute()).copy()

    limit = 0.5
    clamped_solver = DiffIKSolver(handles, DiffIKConfig(max_angvel=limit))
    clamped = clamped_solver.solve(clamped_solver.pose_error.compute()).copy()

    assert np.abs(raw).max() > limit
    assert np.abs(clamped).max() <= limit + 1e-12
    cosine = np.dot(raw, clamped) / (np.linalg.norm(raw) * np.linalg.norm(clamped))
    assert cosine == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# Convergence
# --------------------------------------------------------------------------


def test_near_target_converges_within_200_iterations(handles: RobotHandles) -> None:
    solver = DiffIKSolver(handles, DiffIKConfig())
    _place_target(handles, NEAR_TARGET_OFFSET)
    _iterate_kinematically(solver, 200)

    position_error, orientation_error = _error_norms(solver)
    assert position_error < 1e-3
    assert orientation_error < 1e-3


def test_near_target_converges_in_a_handful_of_iterations(
    handles: RobotHandles,
) -> None:
    """The 200-iteration budget is loose. Newton steps on a smooth problem
    should be done long before that, and a regression that slows convergence
    down would otherwise hide inside the budget."""
    solver = DiffIKSolver(handles, DiffIKConfig())
    _place_target(handles, NEAR_TARGET_OFFSET)
    _iterate_kinematically(solver, 5)

    position_error, orientation_error = _error_norms(solver)
    assert position_error < 1e-6
    assert orientation_error < 1e-6


# --------------------------------------------------------------------------
# Unreachable target
# --------------------------------------------------------------------------


def test_unreachable_target_gives_a_finite_solution(handles: RobotHandles) -> None:
    solver = DiffIKSolver(handles, DiffIKConfig())
    model_mod.reset_to_home(handles)
    model_mod.sync_target_to_site(handles)
    handles.data.mocap_pos[handles.mocap_id] = UNREACHABLE_TARGET

    dq = solver.solve(solver.pose_error.compute())
    assert np.all(np.isfinite(dq))


def test_unreachable_target_does_not_diverge_under_simulation(
    handles: RobotHandles,
) -> None:
    """The arm should stretch toward the target and stop against its joint
    limits, not produce NaN or run away."""
    solver = DiffIKSolver(handles, DiffIKConfig())
    model_mod.reset_to_home(handles)
    model_mod.sync_target_to_site(handles)
    handles.data.mocap_pos[handles.mocap_id] = UNREACHABLE_TARGET

    _simulate(solver, 500)

    assert np.all(np.isfinite(handles.data.qpos))
    assert np.all(np.isfinite(solver.solve(solver.pose_error.compute())))
    arm = handles.data.qpos[handles.qpos_ids]
    assert np.all(arm >= solver.lower_limits - 1e-3)
    assert np.all(arm <= solver.upper_limits + 1e-3)


def test_damping_bounds_the_velocity_near_a_singularity(
    handles: RobotHandles,
) -> None:
    """Driving at an unreachable target pushes the arm to full extension, which
    is where the undamped pseudo-inverse would blow up. Raising the damping has
    to bring the peak joint velocity down."""
    peaks = []
    for damping in (1e-4, 1e-2):
        solver = DiffIKSolver(handles, DiffIKConfig(damping=damping))
        model_mod.reset_to_home(handles)
        model_mod.sync_target_to_site(handles)
        handles.data.mocap_pos[handles.mocap_id] = UNREACHABLE_TARGET

        peak = 0.0
        for _ in range(300):
            solver.step()
            mujoco.mj_step(handles.model, handles.data)
            peak = max(peak, float(np.abs(solver.solve(solver.pose_error.value)).max()))
        peaks.append(peak)

    assert np.all(np.isfinite(peaks))
    assert peaks[1] < peaks[0] / 10.0


# --------------------------------------------------------------------------
# Actuation
# --------------------------------------------------------------------------


def test_step_writes_ctrl_and_leaves_qpos_alone(handles: RobotHandles) -> None:
    """Writing qpos would teleport the arm and hide the convergence behaviour
    the project is meant to show."""
    solver = DiffIKSolver(handles, DiffIKConfig())
    _place_target(handles, NEAR_TARGET_OFFSET)

    qpos_before = handles.data.qpos.copy()
    ctrl_before = handles.data.ctrl.copy()
    solver.step()

    assert np.array_equal(handles.data.qpos, qpos_before)
    assert not np.array_equal(handles.data.ctrl, ctrl_before)


def test_step_leaves_the_gripper_command_untouched(handles: RobotHandles) -> None:
    """actuator8 drives the fingers and is not part of the IK command."""
    solver = DiffIKSolver(handles, DiffIKConfig())
    _place_target(handles, NEAR_TARGET_OFFSET)

    gripper_id = handles.model.nu - 1
    assert gripper_id not in handles.act_ids
    before = handles.data.ctrl[gripper_id]
    solver.step()
    assert handles.data.ctrl[gripper_id] == before


def test_commands_stay_inside_the_joint_limits(handles: RobotHandles) -> None:
    solver = DiffIKSolver(handles, DiffIKConfig())
    model_mod.reset_to_home(handles)
    model_mod.sync_target_to_site(handles)
    handles.data.mocap_pos[handles.mocap_id] = UNREACHABLE_TARGET

    for _ in range(300):
        solver.step()
        mujoco.mj_step(handles.model, handles.data)
        command = handles.data.ctrl[handles.act_ids]
        assert np.all(command >= solver.lower_limits - 1e-9)
        assert np.all(command <= solver.upper_limits + 1e-9)


def test_clipping_is_reported(handles: RobotHandles) -> None:
    """The benchmark counts clipped steps, so the flag has to be honest in both
    directions."""
    solver = DiffIKSolver(handles, DiffIKConfig())
    _place_target(handles, NEAR_TARGET_OFFSET)
    solver.step()
    assert solver.clipped_last_step is False

    model_mod.reset_to_home(handles)
    model_mod.sync_target_to_site(handles)
    handles.data.mocap_pos[handles.mocap_id] = UNREACHABLE_TARGET
    clipped = False
    for _ in range(300):
        solver.step()
        mujoco.mj_step(handles.model, handles.data)
        clipped |= solver.clipped_last_step
    assert clipped is True


# --------------------------------------------------------------------------
# Closed-loop behaviour and its steady-state offset
# --------------------------------------------------------------------------


def test_closed_loop_tracking_settles_and_stays_settled(
    handles: RobotHandles,
) -> None:
    """Under physics the error settles to a small constant rather than to zero,
    and it must not creep afterwards."""
    solver = DiffIKSolver(handles, DiffIKConfig())
    _place_target(handles, NEAR_TARGET_OFFSET)

    _simulate(solver, 1000)
    settled = _error_norms(solver)
    _simulate(solver, 3000)
    later = _error_norms(solver)

    assert settled[0] < 1e-2 and settled[1] < 2e-2
    assert later[0] <= settled[0] * 1.01
    assert later[1] <= settled[1] * 1.01


def test_steady_state_offset_is_gravity_droop(handles: RobotHandles) -> None:
    """The residual is not a solver defect. The position actuators are a PD
    pair: holding against gravity needs a permanent setpoint offset, and that
    offset is exactly what the residual task-space error produces.

    Two consequences follow, and both are checked here: the residual vanishes
    without gravity, and it scales as 1 / integration_dt, because a larger gain
    buys the same actuator force from a smaller error.
    """
    model = handles.model
    original_gravity = model.opt.gravity.copy()
    try:
        model.opt.gravity[:] = 0.0
        solver = DiffIKSolver(handles, DiffIKConfig())
        _place_target(handles, NEAR_TARGET_OFFSET)
        _simulate(solver, 2000)
        assert _error_norms(solver)[0] < 1e-9

        model.opt.gravity[:] = original_gravity
        residuals = []
        for integration_dt in (1.0, 10.0):
            solver = DiffIKSolver(
                handles, DiffIKConfig(integration_dt=integration_dt)
            )
            _place_target(handles, NEAR_TARGET_OFFSET)
            _simulate(solver, 2000)
            residuals.append(_error_norms(solver)[0])

        assert residuals[0] / residuals[1] == pytest.approx(10.0, rel=0.15)
    finally:
        model.opt.gravity[:] = original_gravity


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_config_defaults_match_the_specification() -> None:
    config = DiffIKConfig()
    assert config.damping == 1e-4
    assert config.integration_dt == 1.0
    assert config.nullspace_gain == 0.0
    assert config.max_angvel == 0.0


def test_config_is_frozen() -> None:
    config = DiffIKConfig()
    with pytest.raises(Exception):
        config.damping = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"damping": 0.0},
        {"damping": -1e-3},
        {"integration_dt": 0.0},
        {"max_angvel": -1.0},
        {"home_posture": (0.0, 0.0)},
    ],
)
def test_config_rejects_invalid_values(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        DiffIKConfig(**kwargs)


def test_home_posture_overrides_the_keyframe_reference(
    handles: RobotHandles,
) -> None:
    posture = tuple(float(v) for v in np.zeros(7))
    solver = DiffIKSolver(handles, DiffIKConfig(home_posture=posture))
    assert np.allclose(solver.q_reference, 0.0)

    default_solver = DiffIKSolver(handles, DiffIKConfig())
    assert np.allclose(default_solver.q_reference, handles.q_home)
