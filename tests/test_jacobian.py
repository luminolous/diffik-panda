"""Site Jacobian against finite differences.

This is the foundation test. mj_jacSite fills a 3 x nv row block spanning the
whole model, and the arm occupies only 7 of those columns. If dof_ids picks the
wrong ones the solver still runs and still produces plausible-looking motion, so
nothing downstream catches the mistake. Finite differences do.
"""

import mujoco
import numpy as np
import pytest
from numpy.typing import NDArray

from diffik.model import RobotHandles

# Perturbation used for the finite difference. Small enough that the linear
# term dominates, large enough to stay well above float64 noise in site_xpos.
EPS = 1e-6
TOL = 1e-4

# A handful of configurations, so the check is not accidentally passing at one
# convenient posture. Written out rather than sampled: the test must be
# reproducible without a seed.
TEST_CONFIGURATIONS: tuple[tuple[float, ...], ...] = (
    (0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853),  # home keyframe
    (0.3, -0.4, 0.2, -1.9, 0.5, 1.2, -0.3),
    (-0.8, 0.6, -0.5, -0.9, -0.7, 2.1, 0.9),
    (1.2, 0.9, 1.0, -0.3, 1.4, 3.0, 1.8),  # close to full extension
)


def _set_configuration(
    handles: RobotHandles, q_arm: NDArray[np.float64]
) -> None:
    handles.data.qpos[handles.qpos_ids] = q_arm
    mujoco.mj_forward(handles.model, handles.data)


def _site_pose(handles: RobotHandles) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Current site position and orientation quaternion, as copies."""
    pos = handles.data.site_xpos[handles.site_id].copy()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, handles.data.site_xmat[handles.site_id])
    return pos, quat


def _finite_difference_jacobian(
    handles: RobotHandles, q_arm: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Numerical 6x7 site Jacobian by central differences.

    Central rather than forward: the truncation error drops from O(EPS) to
    O(EPS^2), which keeps the comparison comfortably inside the tolerance
    instead of sitting on top of it.

    The rotational block is the rotation vector of the relative quaternion
    divided by the step, which is the same quantity mj_jacSite reports in jacr:
    an angular velocity, not a quaternion difference.
    """
    jac = np.zeros((6, 7))
    for column in range(7):
        q_plus = q_arm.copy()
        q_plus[column] += EPS
        _set_configuration(handles, q_plus)
        pos_plus, quat_plus = _site_pose(handles)

        q_minus = q_arm.copy()
        q_minus[column] -= EPS
        _set_configuration(handles, q_minus)
        pos_minus, quat_minus = _site_pose(handles)

        jac[:3, column] = (pos_plus - pos_minus) / (2.0 * EPS)

        quat_minus_inv = np.zeros(4)
        relative = np.zeros(4)
        rotation = np.zeros(3)
        mujoco.mju_negQuat(quat_minus_inv, quat_minus)
        mujoco.mju_mulQuat(relative, quat_plus, quat_minus_inv)
        mujoco.mju_quat2Vel(rotation, relative, 1.0)
        jac[3:, column] = rotation / (2.0 * EPS)

    _set_configuration(handles, q_arm)
    return jac


def _analytic_jacobian(handles: RobotHandles) -> NDArray[np.float64]:
    """mj_jacSite restricted to the arm columns."""
    model, data = handles.model, handles.data
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, handles.site_id)
    return np.vstack([jacp[:, handles.dof_ids], jacr[:, handles.dof_ids]])


@pytest.mark.parametrize("configuration", TEST_CONFIGURATIONS)
def test_analytic_jacobian_matches_finite_differences(
    handles: RobotHandles, configuration: tuple[float, ...]
) -> None:
    q_arm = np.asarray(configuration, dtype=np.float64)
    _set_configuration(handles, q_arm)

    analytic = _analytic_jacobian(handles)
    numeric = _finite_difference_jacobian(handles, q_arm)

    per_column = np.linalg.norm(analytic - numeric, axis=0)
    worst = int(np.argmax(per_column))
    assert per_column[worst] < TOL, (
        f"column {worst} disagrees by {per_column[worst]:.3e}; "
        "dof_ids is most likely wrong"
    )


def test_jacobian_has_the_expected_shape(handles: RobotHandles) -> None:
    assert _analytic_jacobian(handles).shape == (6, 7)


def test_arm_columns_are_a_strict_subset_of_the_model_jacobian(
    handles: RobotHandles,
) -> None:
    """The model carries more dofs than the arm. Slicing has to actually drop
    something, otherwise dof_ids was built from the wrong space."""
    assert handles.model.nv > len(handles.dof_ids)


def test_finger_columns_do_not_move_the_site(handles: RobotHandles) -> None:
    """The fingers translate along the gripper, they do not move the hand. Their
    Jacobian columns must be zero, which is why they are excluded from dof_ids
    rather than merely ignored."""
    model, data = handles.model, handles.data
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, handles.site_id)

    finger_dofs = [
        model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in ("finger_joint1", "finger_joint2")
    ]
    assert np.allclose(jacp[:, finger_dofs], 0.0)
    assert np.allclose(jacr[:, finger_dofs], 0.0)
