"""Six-dimensional task-space error between the site and the mocap target."""

from __future__ import annotations

import mujoco
import numpy as np
from numpy.typing import NDArray

from diffik.model import RobotHandles


class PoseError:
    """Position and orientation error, stacked into one length-6 vector.

    The orientation half has to live in the same space as angular velocity,
    because it is about to be multiplied by a Jacobian that maps joint velocity
    to a twist. That rules out subtracting quaternions elementwise: the result
    of such a subtraction is not a rotation and has no consistent scale. The
    relative rotation is converted to a rotation vector instead.
    """

    def __init__(self, handles: RobotHandles) -> None:
        self._handles = handles

        # Every buffer the computation touches is allocated once here, so
        # compute() can run inside the control loop without allocating.
        self._error = np.zeros(6, dtype=np.float64)
        self._site_quat = np.zeros(4, dtype=np.float64)
        self._site_quat_inv = np.zeros(4, dtype=np.float64)
        self._error_quat = np.zeros(4, dtype=np.float64)

    @property
    def value(self) -> NDArray[np.float64]:
        """The buffer filled by the last compute() call."""
        return self._error

    def compute(self) -> NDArray[np.float64]:
        """Return the current 6-DoF error. The array is reused between calls;
        copy it if you need to keep a value across steps."""
        handles = self._handles
        data = handles.data

        # Position: a plain difference, target minus current.
        np.subtract(
            data.mocap_pos[handles.mocap_id],
            data.site_xpos[handles.site_id],
            out=self._error[:3],
        )

        # Orientation: target * conjugate(current) is the rotation that takes
        # the site to the target. The order matters; reversing it gives the
        # rotation expressed in the wrong frame and the arm turns the wrong way.
        mujoco.mju_mat2Quat(self._site_quat, data.site_xmat[handles.site_id])
        mujoco.mju_negQuat(self._site_quat_inv, self._site_quat)
        mujoco.mju_mulQuat(
            self._error_quat,
            data.mocap_quat[handles.mocap_id],
            self._site_quat_inv,
        )
        # Passing 1.0 as the timestep makes this an axis-angle rotation vector
        # rather than a velocity.
        mujoco.mju_quat2Vel(self._error[3:], self._error_quat, 1.0)

        return self._error
