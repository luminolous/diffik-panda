"""Scripted benchmark trajectory.

Every recorded number depends on this path being the same one every time, so
its shape is pinned here rather than left to the benchmark scripts.
"""

import mujoco
import numpy as np
import pytest

from diffik import model as model_mod
from diffik.model import RobotHandles
from diffik.trajectory import SHOULDER, ReachTrajectory


def _radius(position: np.ndarray) -> float:
    return float(np.linalg.norm(position - SHOULDER))


def test_path_starts_and_ends_on_the_home_pose(handles: RobotHandles) -> None:
    """A path that did not close would leave the return leg measuring a
    different pose than the outbound one."""
    trajectory = ReachTrajectory.from_home(handles)
    assert np.allclose(trajectory.position_at(0.0), trajectory.origin)
    assert np.allclose(
        trajectory.position_at(trajectory.duration), trajectory.origin, atol=1e-12
    )


def test_path_reaches_the_peak_radius_halfway(handles: RobotHandles) -> None:
    trajectory = ReachTrajectory.from_home(handles)
    midpoint = trajectory.position_at(trajectory.duration / 2.0)
    assert _radius(midpoint) == pytest.approx(trajectory.peak_radius, abs=1e-12)


def test_radius_never_exceeds_the_peak(handles: RobotHandles) -> None:
    trajectory = ReachTrajectory.from_home(handles)
    radii = [
        _radius(trajectory.position_at(t))
        for t in np.linspace(0.0, trajectory.duration, 200)
    ]
    assert max(radii) <= trajectory.peak_radius + 1e-12
    assert min(radii) >= trajectory.start_radius - 1e-12


def test_path_starts_and_ends_at_rest(handles: RobotHandles) -> None:
    """The raised-cosine profile has zero slope at both ends. A step change
    there would show up as a velocity spike unrelated to conditioning and
    pollute the peak |dq| every benchmark records."""
    trajectory = ReachTrajectory.from_home(handles)
    step = 1e-4
    start_speed = (
        np.linalg.norm(trajectory.position_at(step) - trajectory.position_at(0.0))
        / step
    )
    end_speed = (
        np.linalg.norm(
            trajectory.position_at(trajectory.duration)
            - trajectory.position_at(trajectory.duration - step)
        )
        / step
    )
    assert start_speed < 1e-3
    assert end_speed < 1e-3


def test_time_outside_the_run_is_clamped(handles: RobotHandles) -> None:
    trajectory = ReachTrajectory.from_home(handles)
    assert np.allclose(trajectory.position_at(-5.0), trajectory.origin)
    assert np.allclose(
        trajectory.position_at(trajectory.duration + 5.0),
        trajectory.origin,
        atol=1e-12,
    )


def test_apply_writes_the_target_and_leaves_zero_error_at_the_start(
    handles: RobotHandles,
) -> None:
    trajectory = ReachTrajectory.from_home(handles)
    model_mod.reset_to_home(handles)
    trajectory.apply(handles, 0.0)
    mujoco.mj_forward(handles.model, handles.data)

    assert np.allclose(
        handles.data.mocap_pos[handles.mocap_id], trajectory.origin
    )
    assert np.allclose(
        handles.data.site_xpos[handles.site_id], trajectory.origin, atol=1e-12
    )


def test_step_count_matches_the_duration(handles: RobotHandles) -> None:
    trajectory = ReachTrajectory.from_home(handles)
    timestep = handles.model.opt.timestep
    assert trajectory.steps(timestep) == round(trajectory.duration / timestep)


def test_trajectory_is_deterministic(handles: RobotHandles) -> None:
    first = ReachTrajectory.from_home(handles)
    second = ReachTrajectory.from_home(handles)
    times = np.linspace(0.0, first.duration, 50)
    assert np.allclose(
        [first.position_at(t) for t in times],
        [second.position_at(t) for t in times],
    )
