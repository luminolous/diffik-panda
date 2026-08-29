"""Damped least squares differential inverse kinematics."""

from __future__ import annotations

import mujoco
import numpy as np
from numpy.typing import NDArray

from diffik.config import DiffIKConfig
from diffik.error import PoseError
from diffik.model import ARM_JOINT_NAMES, RobotHandles

_ARM_DOF = 7
_TASK_DIM = 6


class DiffIKSolver:
    """One linear solve per control step, converging over time.

    The Jacobian maps joint velocity to end-effector twist, `xdot = J qdot`.
    Inverting that for qdot is the whole problem, and the obvious inverse is the
    pseudo-inverse `J^+ = J^T (J J^T)^-1`. It blows up whenever J loses rank:
    the arm near full extension, or two joint axes lining up. The joint velocity
    goes to infinity for a finite task-space error.

    Damped least squares replaces it with

        qdot = J^T (J J^T + lambda^2 I)^-1 e

    The added lambda^2 I keeps the 6x6 matrix invertible no matter what J does.
    Away from singularities the extra term is negligible and the result matches
    the pseudo-inverse; near one it bounds the joint velocity at the cost of
    tracking accuracy. That trade is the point of the method.
    """

    def __init__(self, handles: RobotHandles, config: DiffIKConfig) -> None:
        self.handles = handles
        self.config = config
        self.pose_error = PoseError(handles)

        model = handles.model
        nv = model.nv

        # Joint limits, resolved once. The names are looked up here rather than
        # in step() so the control loop stays free of mj_name2id.
        lower = np.empty(_ARM_DOF, dtype=np.float64)
        upper = np.empty(_ARM_DOF, dtype=np.float64)
        for i, name in enumerate(ARM_JOINT_NAMES):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if model.jnt_limited[joint_id]:
                lower[i], upper[i] = model.jnt_range[joint_id]
            else:
                lower[i], upper[i] = -np.inf, np.inf
        self.lower_limits = lower
        self.upper_limits = upper

        # Posture the nullspace term pulls toward.
        if config.home_posture is None:
            self.q_reference = handles.q_home.copy()
        else:
            self.q_reference = np.asarray(config.home_posture, dtype=np.float64)

        # Working buffers. mj_jacSite writes the linear block into the first
        # three rows and the angular block into the last three, so one 6 x nv
        # array with two views over it avoids stacking them every step.
        self._jac_full = np.zeros((_TASK_DIM, nv), dtype=np.float64)
        self._jacp = self._jac_full[:3]
        self._jacr = self._jac_full[3:]

        self._jac = np.zeros((_TASK_DIM, _ARM_DOF), dtype=np.float64)
        self._jjt = np.zeros((_TASK_DIM, _TASK_DIM), dtype=np.float64)
        self._damping_eye = np.eye(_TASK_DIM, dtype=np.float64)
        self._dq = np.zeros(_ARM_DOF, dtype=np.float64)
        self._qdot_secondary = np.zeros(_ARM_DOF, dtype=np.float64)
        self._q_current = np.zeros(_ARM_DOF, dtype=np.float64)
        self._task_velocity = np.zeros(_TASK_DIM, dtype=np.float64)
        self._dq_full = np.zeros(nv, dtype=np.float64)
        self._q = np.zeros(model.nq, dtype=np.float64)
        self._q_arm = np.zeros(_ARM_DOF, dtype=np.float64)
        self._q_unclipped = np.zeros(_ARM_DOF, dtype=np.float64)

        self.clipped_last_step = False
        """Whether joint-limit clipping changed the command on the last step.
        The benchmark counts these: clipping alters the direction of motion,
        which is exactly what a QP formulation avoids."""

        self.clipped_joints = np.zeros(_ARM_DOF, dtype=bool)
        """Which joints the clip touched on the last step. Reused between steps;
        copy it to keep a value. A run-level count says that clipping happened,
        this says where, which is what separates one joint jamming from the
        whole arm saturating."""

        self.clip_overshoot = np.zeros(_ARM_DOF, dtype=np.float64)
        """How far past its limit each joint's setpoint reached before the clip,
        in radians, signed. Reused between steps."""

    @property
    def joint_velocity(self) -> NDArray[np.float64]:
        """Joint velocity produced by the last solve. Exposed for the
        benchmarks, which record its peak; the buffer is reused, so copy it if
        you need to keep the value."""
        return self._dq

    def jacobian(self) -> NDArray[np.float64]:
        """Current 6x7 site Jacobian, restricted to the arm columns.

        mj_jacSite fills all nv columns, including the two gripper dofs. Taking
        the arm columns by index is not cosmetic: leaving the finger columns in
        would add two all-zero directions to J J^T and make the damping carry
        them.
        """
        handles = self.handles
        mujoco.mj_jacSite(
            handles.model, handles.data, self._jacp, self._jacr, handles.site_id
        )
        np.take(self._jac_full, handles.dof_ids, axis=1, out=self._jac)
        return self._jac

    def solve(self, error: NDArray[np.float64]) -> NDArray[np.float64]:
        """Joint velocity that reduces the given 6-DoF error.

        Written as `J^T solve(J J^T + lambda^2 I, e)`, never as `pinv(J) @ e`.
        The matrix being inverted is only 6x6, and forming the pseudo-inverse
        explicitly would throw away the damping that makes this stable.

        The returned array is reused between calls.
        """
        jac = self.jacobian()

        np.matmul(jac, jac.T, out=self._jjt)
        self._jjt += self.config.damping**2 * self._damping_eye

        # np.linalg.solve allocates its 6-element result; every other buffer in
        # this method is preallocated.
        np.matmul(jac.T, np.linalg.solve(self._jjt, error), out=self._dq)

        self._add_nullspace_term(jac)
        self._limit_angular_velocity()
        return self._dq

    def _add_nullspace_term(self, jac: NDArray[np.float64]) -> None:
        """Add a secondary objective that cannot disturb the primary task.

        Seven joints serve a six-dimensional task, so one direction of joint
        motion leaves the end-effector where it is: the elbow can swing while
        the gripper stays put. Projecting a desired secondary velocity onto that
        direction spends the spare freedom for free.

            qdot = J^+ e + (I - J^+ J) qdot_0,  qdot_0 = Kn (q_home - q)

        The projector is never formed. Applying it to a vector is

            (I - J^+ J) v = v - J^T (J J^T + lambda^2 I)^-1 (J v)

        which reuses the matrix already factored for the primary solve, stays a
        6x6 problem instead of a 7x7 one, and keeps the same damping on both
        terms. Building J^+ explicitly with pinv would drop that damping and
        reintroduce the blow-up near singularities that the primary solve is
        written to avoid.
        """
        gain = self.config.nullspace_gain
        if gain == 0.0:
            return

        handles = self.handles
        np.take(handles.data.qpos, handles.qpos_ids, out=self._q_current)
        np.subtract(self.q_reference, self._q_current, out=self._qdot_secondary)
        self._qdot_secondary *= gain

        np.matmul(jac, self._qdot_secondary, out=self._task_velocity)
        self._qdot_secondary -= jac.T @ np.linalg.solve(
            self._jjt, self._task_velocity
        )
        self._dq += self._qdot_secondary

    def _limit_angular_velocity(self) -> None:
        """Scale dq down so max(|dq|) stays under the configured bound.

        Scaling the whole vector keeps its direction. Clamping each entry
        separately would bend the motion into a different Cartesian direction.
        """
        max_angvel = self.config.max_angvel
        if max_angvel <= 0.0:
            return
        peak = np.abs(self._dq).max()
        if peak > max_angvel:
            self._dq *= max_angvel / peak

    def step(self) -> None:
        """One control step: error, solve, integrate, write ctrl.

        Nothing here touches data.qpos. Writing configuration directly would
        teleport the arm and hide the convergence behaviour this project is
        meant to show. The Panda ships with position actuators that run their
        own PD loop, so the command is a setpoint, not a torque.
        """
        handles = self.handles
        model, data = handles.model, handles.data

        error = self.pose_error.compute()
        dq = self.solve(error)

        # dq covers the arm only; mj_integratePos expects a full nv vector, and
        # the remaining entries must stay zero so the gripper does not drift.
        self._dq_full[handles.dof_ids] = dq

        np.copyto(self._q, data.qpos)
        mujoco.mj_integratePos(model, self._q, self._dq_full, self.config.integration_dt)

        # Fancy indexing returns a copy, so the clip is done on an explicit
        # buffer and the result is written to ctrl. Clipping in place through
        # self._q[qpos_ids] would silently discard the result.
        np.take(self._q, handles.qpos_ids, out=self._q_unclipped)

        if self.config.clip_joint_limits:
            np.clip(
                self._q_unclipped, self.lower_limits, self.upper_limits, out=self._q_arm
            )
            # Per-joint record of what the clip removed. The sign says which way
            # the setpoint was heading: positive means it was pushed back down
            # from the upper limit, negative up from the lower one.
            np.subtract(self._q_unclipped, self._q_arm, out=self.clip_overshoot)
            np.not_equal(self.clip_overshoot, 0.0, out=self.clipped_joints)
        else:
            np.copyto(self._q_arm, self._q_unclipped)
            self.clip_overshoot.fill(0.0)
            self.clipped_joints.fill(False)

        self.clipped_last_step = bool(self.clipped_joints.any())

        data.ctrl[handles.act_ids] = self._q_arm
