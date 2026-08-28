"""Scripted target trajectory shared by every benchmark and recording.

Keeping it here rather than in a script is what makes the benchmark numbers
comparable: the damped least squares sweep, the mink comparison and the demo
recording all drive the same path over the same duration.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

from diffik.model import RobotHandles

# The arm hangs off link1 at this height, so radii are measured from here
# rather than from the world origin.
SHOULDER = np.array([0.0, 0.0, 0.333])

# The Panda reaches about 0.85 m from the shoulder when only the position is
# constrained, but this path holds the orientation fixed as well, and that costs
# a lot of workspace: past roughly 0.70 m the arm can no longer keep the home
# orientation and the pose becomes infeasible rather than merely ill-conditioned.
#
# 0.70 puts the run right on that edge, which is what makes the sweep readable.
# Away from it every damping value below 1e-1 tracks identically, and at the
# edge cond(J) reaches the tens of thousands, so the small values spike |dq| and
# start clipping while the large ones stay smooth. Pushing further, to 0.72 or
# beyond, replaces that contrast with total tracking failure at every damping
# value, which measures nothing.
DEFAULT_PEAK_RADIUS = 0.70
DEFAULT_DURATION = 6.0


@dataclass(frozen=True)
class ReachTrajectory:
    """Target that travels straight out from the home pose toward the edge of
    the orientation-constrained workspace and comes back, on a raised-cosine
    profile.

    The profile starts and ends with zero velocity, so the run contains no step
    change that would show up as a spike unrelated to conditioning. Orientation
    is held at the home value throughout: the interesting variable is how close
    the arm is to a singularity, and a rotating target would mix a second effect
    into the same numbers.
    """

    origin: NDArray[np.float64]
    """Site position at the home keyframe, where the path starts and ends."""

    quaternion: NDArray[np.float64]
    """Site orientation at the home keyframe, held constant."""

    start_radius: float
    peak_radius: float = DEFAULT_PEAK_RADIUS
    duration: float = DEFAULT_DURATION

    @classmethod
    def from_home(
        cls,
        handles: RobotHandles,
        peak_radius: float = DEFAULT_PEAK_RADIUS,
        duration: float = DEFAULT_DURATION,
    ) -> "ReachTrajectory":
        """Build the path from the model's own home pose, so it does not depend
        on hardcoded coordinates that would silently drift if the scene changed."""
        from diffik import model as model_mod

        model_mod.reset_to_home(handles)
        origin = handles.data.site_xpos[handles.site_id].copy()
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, handles.data.site_xmat[handles.site_id])

        return cls(
            origin=origin,
            quaternion=quaternion,
            start_radius=float(np.linalg.norm(origin - SHOULDER)),
            peak_radius=peak_radius,
            duration=duration,
        )

    @property
    def direction(self) -> NDArray[np.float64]:
        """Unit vector from the shoulder toward the home pose."""
        offset = self.origin - SHOULDER
        return offset / np.linalg.norm(offset)

    def position_at(self, time: float) -> NDArray[np.float64]:
        """Target position at a given time, clamped to the run duration."""
        phase = np.clip(time / self.duration, 0.0, 1.0)
        # Raised cosine: 0 at both ends, 1 halfway, zero slope at 0 and 1.
        blend = 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))
        radius = self.start_radius + (self.peak_radius - self.start_radius) * blend
        return SHOULDER + self.direction * radius

    def apply(self, handles: RobotHandles, time: float) -> None:
        """Write the target pose for this time into the mocap body."""
        handles.data.mocap_pos[handles.mocap_id] = self.position_at(time)
        handles.data.mocap_quat[handles.mocap_id] = self.quaternion

    def steps(self, timestep: float) -> int:
        return int(round(self.duration / timestep))
