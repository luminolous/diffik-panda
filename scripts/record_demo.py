"""Render the tracking demo headless and write media/tracking.gif.

Drives the same trajectory the benchmarks use, so the animation shows the run
the numbers describe rather than a separate hand-made motion.

    python scripts/record_demo.py

No viewer window is opened; frames come from mujoco.Renderer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from diffik import model as robot_model  # noqa: E402
from diffik.config import DiffIKConfig  # noqa: E402
from diffik.solver import DiffIKSolver  # noqa: E402
from diffik.trajectory import ReachTrajectory  # noqa: E402

DEFAULT_SCENE = REPO_ROOT / "scene" / "panda_ik.xml"
DEFAULT_OUTPUT = REPO_ROOT / "media" / "tracking.gif"

# The damping sweep puts this value in the clean regime: no clipping, lowest
# tracking error. A recording made in the failure band would show the arm stuck
# against a limit, which is a useful picture but not the one a README opens with.
DEMO_DAMPING = 1e-2
DEMO_NULLSPACE_GAIN = 5.0

# Framing chosen so the base, the elbow and the target all stay in shot for the
# whole reach.
CAMERA_AZIMUTH = 150.0
CAMERA_ELEVATION = -15.0
CAMERA_DISTANCE = 1.3
CAMERA_LOOKAT = (0.35, 0.0, 0.50)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="playback rate; frames are sampled to keep the motion real time",
    )
    parser.add_argument(
        "--colors",
        type=int,
        default=128,
        help="palette size per frame, at most 256 (default: %(default)s)",
    )
    parser.add_argument("--damping", type=float, default=DEMO_DAMPING)
    parser.add_argument(
        "--nullspace-gain", type=float, default=DEMO_NULLSPACE_GAIN
    )
    return parser.parse_args(argv)


def build_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = CAMERA_AZIMUTH
    camera.elevation = CAMERA_ELEVATION
    camera.distance = CAMERA_DISTANCE
    camera.lookat[:] = CAMERA_LOOKAT
    return camera


def record(args: argparse.Namespace) -> list[np.ndarray]:
    handles = robot_model.load(args.scene)
    model, data = handles.model, handles.data
    solver = DiffIKSolver(
        handles,
        DiffIKConfig(damping=args.damping, nullspace_gain=args.nullspace_gain),
    )
    trajectory = ReachTrajectory.from_home(handles)

    robot_model.reset_to_home(handles)
    robot_model.sync_target_to_site(handles)

    # Sample every nth physics step so the GIF plays at real time.
    stride = max(1, round(1.0 / (args.fps * model.opt.timestep)))
    camera = build_camera()

    frames: list[np.ndarray] = []
    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        for step in range(trajectory.steps(model.opt.timestep)):
            trajectory.apply(handles, data.time)
            mujoco.mj_forward(model, data)
            solver.step()
            mujoco.mj_step(model, data)

            if step % stride == 0:
                renderer.update_scene(data, camera)
                frames.append(renderer.render())

    return frames


def write_gif(
    frames: list[np.ndarray], output: Path, fps: int, colors: int
) -> None:
    """Encode the frames, quantising each one to an adaptive palette first.

    A GIF holds at most 256 colours per frame anyway, so the palette is not a
    quality decision that can be skipped, only one that can be made badly.
    Handing raw RGB frames to a naive writer produces a file about five times
    larger at the same resolution: 8.2 MB against 1.5 MB here.
    """
    images = [
        Image.fromarray(frame).quantize(colors=colors, method=Image.MEDIANCUT)
        for frame in frames
    ]
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=round(1000.0 / fps),
        loop=0,
        optimize=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    frames = record(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_gif(frames, args.output, args.fps, args.colors)

    size_mb = args.output.stat().st_size / 1e6
    print(
        f"wrote {len(frames)} frames at {args.width}x{args.height} to "
        f"{args.output} ({size_mb:.2f} MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
