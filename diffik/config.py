"""Tuning parameters for the differential IK loop."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiffIKConfig:
    """Frozen so a script cannot mutate the tuning halfway through a run and
    leave the recorded benchmark numbers describing two different solvers."""

    damping: float = 1e-4
    """Lambda in the damped least squares solve. Away from singularities the
    result is close to the pseudo-inverse solution; near one it trades tracking
    accuracy for a bounded joint velocity. Raise toward 1e-2 if the arm
    oscillates at its reach limit."""

    integration_dt: float = 1.0
    """Gain, not the physics timestep. dq is a velocity; multiplying it by this
    value extrapolates a position setpoint. Larger tracks more aggressively.
    Deliberately kept separate from model.opt.timestep. Reduce toward 0.1 if the
    motion jitters."""

    nullspace_gain: float = 0.0
    """Strength of the secondary pull toward the home posture, projected into
    the nullspace of the Cartesian task. 0 disables the term."""

    max_angvel: float = 0.0
    """Upper bound on max(|dq|), in rad/s. 0 disables the clamp. Scaling the
    whole vector preserves its direction, unlike the joint-limit clipping that
    happens after integration."""

    clip_joint_limits: bool = True
    """Whether the integrated setpoint is clipped to the joint limits before it
    is written to data.ctrl. Setting this to False is an experiment, not a
    tuning option: it is how the ablation separates a failure caused by clipping
    from one caused by the posture the arm drifted into.

    Turning it off does not let the arm leave its joint range. MuJoCo clamps
    data.ctrl to each actuator's ctrlrange, which for this model equals the
    joint range, and the limit constraints in the solver still hold the arm
    back. What changes is that the setpoint may sit outside the range and keep
    pushing, instead of being redirected along the limit surface."""

    home_posture: tuple[float, ...] | None = None
    """Reference posture for the nullspace term. None means use the home
    keyframe carried by RobotHandles. A tuple rather than an array so the
    dataclass stays immutable and comparable."""

    def __post_init__(self) -> None:
        if self.damping <= 0.0:
            raise ValueError(
                f"damping must be positive, got {self.damping}; a zero damping "
                "is the undamped pseudo-inverse this project exists to avoid"
            )
        if self.integration_dt <= 0.0:
            raise ValueError(
                f"integration_dt must be positive, got {self.integration_dt}"
            )
        if self.max_angvel < 0.0:
            raise ValueError(
                f"max_angvel must be non-negative, got {self.max_angvel}"
            )
        if self.home_posture is not None and len(self.home_posture) != 7:
            raise ValueError(
                f"home_posture must have 7 entries, got {len(self.home_posture)}"
            )
