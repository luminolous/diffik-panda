"""Interactive demo: drag the red box, the gripper follows it.

Run it from the repository root:

    python scripts/run_viewer.py

Drag the target in the viewer with Ctrl + right-click. Ctrl + left-click
rotates it. Press Backspace to reset the arm and put the target back on the
gripper.

Seeing what --nullspace-gain does takes the right experiment. The arm starts at
the home posture and the secondary objective pulls toward that same posture, so
a short drag shows nothing: there is nothing to correct, and gain 0 and gain 5
differ by about 0.02 rad.

The term earns its place by stopping drift. Drag the target on a wide loop
around the workspace and bring it back to where it started, then compare:

    python scripts/run_viewer.py --nullspace-gain 0
    python scripts/run_viewer.py --nullspace-gain 5

With the gain off the elbow keeps whatever configuration the wandering left it
in, ending up around 2.8 rad away from the home posture. With the gain on it
comes back to within 0.01 rad, at the same gripper pose.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from diffik import model as robot_model  # noqa: E402
from diffik.config import DiffIKConfig  # noqa: E402
from diffik.solver import DiffIKSolver  # noqa: E402

DEFAULT_SCENE = REPO_ROOT / "scene" / "panda_ik.xml"

# GLFW keycode. mujoco.viewer forwards raw GLFW codes to the callback, and
# exposes no symbolic names for them.
KEY_BACKSPACE = 259


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=DEFAULT_SCENE,
        help="scene XML to load (default: %(default)s)",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=DiffIKConfig.damping,
        help=(
            "damped least squares lambda. Raise toward 1e-2 if the arm "
            "oscillates near its reach limit (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--integration-dt",
        type=float,
        default=DiffIKConfig.integration_dt,
        help=(
            "gain converting the solved joint velocity into a position "
            "setpoint. Not the physics timestep. Lower it toward 0.1 if the "
            "motion jitters; raise it to shrink the steady-state offset the "
            "position actuators leave under gravity (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--nullspace-gain",
        type=float,
        default=DiffIKConfig.nullspace_gain,
        help=(
            "strength of the secondary pull toward the home posture, projected "
            "into the nullspace of the Cartesian task (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--max-angvel",
        type=float,
        default=DiffIKConfig.max_angvel,
        help=(
            "upper bound on max(|dq|) in rad/s, 0 disables it. Makes the "
            "velocity limit explicit instead of relying on position clipping "
            "(default: %(default)s)"
        ),
    )
    return parser.parse_args(argv)


def build(args: argparse.Namespace) -> tuple[robot_model.RobotHandles, DiffIKSolver]:
    handles = robot_model.load(args.scene)
    config = DiffIKConfig(
        damping=args.damping,
        integration_dt=args.integration_dt,
        nullspace_gain=args.nullspace_gain,
        max_angvel=args.max_angvel,
    )
    return handles, DiffIKSolver(handles, config)


def run(handles: robot_model.RobotHandles, solver: DiffIKSolver) -> None:
    model, data = handles.model, handles.data

    robot_model.reset_to_home(handles)
    # Without this the target sits wherever the XML put it and the arm lunges
    # on the first frame.
    robot_model.sync_target_to_site(handles)

    # Backspace is also the viewer's own reset shortcut, handled inside the
    # built-in UI. That reset goes to qpos0, which for the Panda is not even a
    # legal configuration: joint4 is limited to [-3.0718, -0.0698] and qpos0
    # puts it at 0. It also returns the target to its XML pose with an identity
    # quaternion, which sits a full pi away from the site orientation. The arm
    # then chases a half-turn it cannot make, hits its limits, and stays stuck.
    #
    # Rather than race the built-in handler, the callback only raises a flag and
    # the loop applies the reset on the next iteration, once key handling for
    # that frame is finished. Whoever ran first, the keyframe reset wins.
    reset_requested = False

    def on_key(keycode: int) -> None:
        nonlocal reset_requested
        if keycode == KEY_BACKSPACE:
            reset_requested = True

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=on_key,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        while viewer.is_running():
            loop_start = time.perf_counter()

            if reset_requested:
                robot_model.reset_to_home(handles)
                robot_model.sync_target_to_site(handles)
                reset_requested = False

            solver.step()
            mujoco.mj_step(model, data)
            viewer.sync()

            # Pace the loop to the simulated timestep, otherwise the arm moves
            # as fast as the machine can integrate and the demo is unreadable.
            remaining = model.opt.timestep - (time.perf_counter() - loop_start)
            if remaining > 0:
                time.sleep(remaining)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    handles, solver = build(args)
    print(
        f"damping={solver.config.damping:g}  "
        f"integration_dt={solver.config.integration_dt:g}  "
        f"nullspace_gain={solver.config.nullspace_gain:g}  "
        f"max_angvel={solver.config.max_angvel:g}"
    )
    print("drag the red box with Ctrl + right-click, Backspace resets")
    run(handles, solver)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
