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


def test_clipping_can_be_disabled(handles: RobotHandles) -> None:
    """With the clip off the setpoint is allowed past the joint range.

    This is what the ablation needs: a run where the commanded direction is
    never redirected along a limit surface. It does not let the arm itself
    leave its range. MuJoCo clamps data.ctrl to each actuator's ctrlrange, which
    equals the joint range for this model, and it does so without rewriting
    data.ctrl, which is why the assertion below can still see the raw value.
    """
    model_mod.reset_to_home(handles)
    model_mod.sync_target_to_site(handles)
    handles.data.mocap_pos[handles.mocap_id] = UNREACHABLE_TARGET

    solver = DiffIKSolver(
        handles, DiffIKConfig(damping=1e-2, clip_joint_limits=False)
    )
    went_out_of_range = False
    for _ in range(300):
        solver.step()
        command = handles.data.ctrl[handles.act_ids]
        went_out_of_range |= bool(
            np.any(command < solver.lower_limits - 1e-9)
            or np.any(command > solver.upper_limits + 1e-9)
        )
        mujoco.mj_step(handles.model, handles.data)

    assert went_out_of_range
    assert np.all(np.isfinite(handles.data.qpos))


def test_disabled_clipping_reports_nothing_clipped(handles: RobotHandles) -> None:
    """The counters have to describe what actually happened, otherwise the
    ablation would credit the clip for steps where it never ran."""
    model_mod.reset_to_home(handles)
    model_mod.sync_target_to_site(handles)
    handles.data.mocap_pos[handles.mocap_id] = UNREACHABLE_TARGET

    solver = DiffIKSolver(
        handles, DiffIKConfig(damping=1e-2, clip_joint_limits=False)
    )
    for _ in range(200):
        solver.step()
        mujoco.mj_step(handles.model, handles.data)
        assert solver.clipped_last_step is False
        assert not solver.clipped_joints.any()
        assert np.all(solver.clip_overshoot == 0.0)


def test_clip_instrumentation_matches_what_the_clip_removed(
    handles: RobotHandles,
) -> None:
    """clip_overshoot must equal the amount taken off the setpoint, per joint,
    with the sign saying which limit was hit."""
    model_mod.reset_to_home(handles)
    model_mod.sync_target_to_site(handles)
    handles.data.mocap_pos[handles.mocap_id] = UNREACHABLE_TARGET

    solver = DiffIKSolver(handles, DiffIKConfig())
    seen_clip = False
    for _ in range(300):
        solver.step()
        command = handles.data.ctrl[handles.act_ids]

        assert np.array_equal(solver.clipped_joints, solver.clip_overshoot != 0.0)
        assert solver.clipped_last_step == bool(solver.clipped_joints.any())

        for joint in np.flatnonzero(solver.clipped_joints):
            seen_clip = True
            overshoot = solver.clip_overshoot[joint]
            if overshoot > 0:
                assert command[joint] == pytest.approx(solver.upper_limits[joint])
            else:
                assert command[joint] == pytest.approx(solver.lower_limits[joint])

        mujoco.mj_step(handles.model, handles.data)

    assert seen_clip, "the unreachable target should have forced a clip"


def test_clip_buffers_are_reused(handles: RobotHandles) -> None:
    solver = DiffIKSolver(handles, DiffIKConfig())
    _place_target(handles, NEAR_TARGET_OFFSET)

    joints, overshoot = solver.clipped_joints, solver.clip_overshoot
    solver.step()
    assert solver.clipped_joints is joints
    assert solver.clip_overshoot is overshoot
    assert joints.shape == (7,) and overshoot.shape == (7,)


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
    assert config.clip_joint_limits is True


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


# --------------------------------------------------------------------------
# Nullspace projection
# --------------------------------------------------------------------------

# Displaces the elbow without being extreme, so the redundant direction has
# something to work with once the primary task is satisfied.
ELBOW_OFFSET = np.array([0.6, 0.0, 0.5, 0.3, 0.4, 0.0, 0.8])

# A gain that is stable under the kinematic iteration used by these tests.
# See test_high_gain_needs_a_smaller_integration_dt for why 5.0 is not.
STABLE_GAIN = 1.0


def _place_target_from_posture(
    handles: RobotHandles, posture_offset: NDArray[np.float64]
) -> None:
    """Start from a displaced posture with the target on the gripper, then move
    the target. The arm has to solve the task and has spare freedom left over."""
    model_mod.reset_to_home(handles)
    handles.data.qpos[handles.qpos_ids] = handles.q_home + posture_offset
    mujoco.mj_forward(handles.model, handles.data)
    model_mod.sync_target_to_site(handles)
    handles.data.mocap_pos[handles.mocap_id] += np.array([0.08, 0.10, -0.05])


def _converged_state(
    handles: RobotHandles, gain: float, integration_dt: float = 1.0
) -> tuple[NDArray[np.float64], NDArray[np.float64], DiffIKSolver]:
    solver = DiffIKSolver(
        handles,
        DiffIKConfig(nullspace_gain=gain, integration_dt=integration_dt),
    )
    _place_target_from_posture(handles, ELBOW_OFFSET)
    _iterate_kinematically(solver, 400)
    return (
        handles.data.qpos[handles.qpos_ids].copy(),
        handles.data.site_xpos[handles.site_id].copy(),
        solver,
    )


def test_zero_gain_disables_the_term_exactly(handles: RobotHandles) -> None:
    """A disabled secondary objective must cost nothing and change nothing, so
    every benchmark run with the term off is comparable to a build without it."""
    _place_target_from_posture(handles, ELBOW_OFFSET)

    plain = DiffIKSolver(handles, DiffIKConfig(nullspace_gain=0.0))
    dq_plain = plain.solve(plain.pose_error.compute()).copy()

    tiny = DiffIKSolver(handles, DiffIKConfig(nullspace_gain=1e-12))
    dq_tiny = tiny.solve(tiny.pose_error.compute()).copy()

    assert np.allclose(dq_plain, dq_tiny, atol=1e-9)


def test_nullspace_component_does_not_move_the_end_effector(
    handles: RobotHandles,
) -> None:
    """The defining property, checked directly rather than through its
    consequences: whatever the secondary objective adds to dq must map to zero
    task-space velocity."""
    _place_target_from_posture(handles, ELBOW_OFFSET)

    plain = DiffIKSolver(handles, DiffIKConfig(nullspace_gain=0.0))
    dq_plain = plain.solve(plain.pose_error.compute()).copy()

    biased = DiffIKSolver(handles, DiffIKConfig(nullspace_gain=STABLE_GAIN))
    dq_biased = biased.solve(biased.pose_error.compute()).copy()

    added = dq_biased - dq_plain
    assert np.linalg.norm(added) > 1e-3, "the secondary objective did nothing"

    twist = biased.jacobian() @ added
    assert np.linalg.norm(twist) < 1e-6


def test_two_gains_reach_the_same_pose_by_different_joint_paths(
    handles: RobotHandles,
) -> None:
    """The milestone criterion: same end-effector pose, different arm."""
    q_off, x_off, _ = _converged_state(handles, 0.0)
    q_on, x_on, _ = _converged_state(handles, STABLE_GAIN)

    assert np.linalg.norm(q_on - q_off) > 1e-2
    assert np.linalg.norm(x_on - x_off) < 1e-4


def test_nullspace_pulls_the_posture_toward_the_reference(
    handles: RobotHandles,
) -> None:
    q_off, _, _ = _converged_state(handles, 0.0)
    q_on, _, _ = _converged_state(handles, STABLE_GAIN)

    assert np.linalg.norm(q_on - handles.q_home) < np.linalg.norm(
        q_off - handles.q_home
    )


def test_primary_task_still_converges_with_the_term_on(
    handles: RobotHandles,
) -> None:
    _, _, solver = _converged_state(handles, STABLE_GAIN)
    position_error, orientation_error = _error_norms(solver)
    assert position_error < 1e-4
    assert orientation_error < 1e-4


def test_high_gain_needs_a_smaller_integration_dt(handles: RobotHandles) -> None:
    """The secondary objective is a proportional controller integrated with an
    explicit step, so it is the product nullspace_gain * integration_dt that has
    to stay small, not the gain alone. At 5.0 and 1.0 the kinematic iteration
    overshoots and grows without bound; the same gain is well behaved once the
    step is shortened.

    The closed loop is more forgiving, because the position actuators add their
    own damping. That is why the viewer tolerates the gain of 5 the
    documentation suggests.
    """
    q_unstable, _, _ = _converged_state(handles, 5.0, integration_dt=1.0)
    assert np.linalg.norm(q_unstable - handles.q_home) > 1e3

    q_stable, _, _ = _converged_state(handles, 5.0, integration_dt=0.3)
    assert np.all(np.isfinite(q_stable))
    assert np.linalg.norm(q_stable - handles.q_home) < 10.0


def test_nullspace_stops_the_posture_drifting_over_a_closed_path(
    handles: RobotHandles,
) -> None:
    """What the term actually buys, in the form a viewer session shows it.

    Starting at home and moving a little proves nothing: the pull points at the
    posture the arm is already in. Send the target on a loop and bring it back
    to where it started, and the difference is the whole point. Without the term
    the arm keeps whatever configuration the wandering left behind; with it the
    arm returns to the home posture at the same gripper pose.
    """
    loop = (
        (0.20, 0.25, 0.10),
        (-0.15, -0.40, 0.20),
        (0.10, 0.30, -0.25),
        (-0.15, -0.15, -0.05),
    )

    def wander(gain: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        solver = DiffIKSolver(handles, DiffIKConfig(nullspace_gain=gain))
        model_mod.reset_to_home(handles)
        model_mod.sync_target_to_site(handles)
        for leg in loop:
            handles.data.mocap_pos[handles.mocap_id] += np.array(leg)
            _simulate(solver, 600)
        return (
            handles.data.qpos[handles.qpos_ids].copy(),
            handles.data.site_xpos[handles.site_id].copy(),
        )

    q_drifted, x_drifted = wander(0.0)
    q_held, x_held = wander(5.0)

    assert np.linalg.norm(q_drifted - handles.q_home) > 1.0
    assert np.linalg.norm(q_held - handles.q_home) < 0.1
    # Same place, different arm.
    assert np.linalg.norm(x_held - x_drifted) < 1e-2


def test_nullspace_follows_the_configured_reference_posture(
    handles: RobotHandles,
) -> None:
    """Moving the reference posture along the redundant direction has to move
    the arm, by exactly the projected amount.

    The reference offsets are built from the actual nullspace of J rather than
    picked by hand. Seven joints against a six-dimensional task leave a single
    redundant direction, and an arbitrary posture offset is almost entirely
    orthogonal to it: the projector removes nearly all of it and two very
    different references produce the same command. Only motion along that one
    direction survives, which is the whole point of the projector.
    """
    _place_target_from_posture(handles, ELBOW_OFFSET)

    probe = DiffIKSolver(handles, DiffIKConfig(nullspace_gain=STABLE_GAIN))
    probe.solve(probe.pose_error.compute())
    # Last right-singular vector of a 6x7 matrix spans its nullspace.
    nullspace_direction = np.linalg.svd(probe.jacobian())[2][-1]
    assert np.linalg.norm(probe.jacobian() @ nullspace_direction) < 1e-9

    q_now = handles.data.qpos[handles.qpos_ids]
    commands = []
    for sign in (+1.0, -1.0):
        reference = tuple(float(v) for v in q_now + sign * nullspace_direction)
        solver = DiffIKSolver(
            handles,
            DiffIKConfig(nullspace_gain=STABLE_GAIN, home_posture=reference),
        )
        commands.append(solver.solve(solver.pose_error.compute()).copy())

    # The projector leaves a nullspace vector untouched, so the two commands
    # differ by exactly twice the gain times that direction.
    difference = commands[0] - commands[1]
    expected = 2.0 * STABLE_GAIN * nullspace_direction
    assert np.allclose(difference, expected, atol=1e-9)
