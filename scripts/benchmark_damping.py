"""Sweep the damping term over a trajectory that runs into a singularity.

The target travels straight out from the home pose until the arm is close to
full extension, then comes back. Every damping value sees the same path, the
same duration and the same timestep, so the runs are comparable.

    python scripts/benchmark_damping.py

Writes one row per simulation step to results/damping_sweep.csv and prints the
per-configuration summary. The per-step rows are what notebooks/02_results.ipynb
plots; the summary is derived from them, never typed in by hand.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from diffik import model as robot_model  # noqa: E402
from diffik.config import DiffIKConfig  # noqa: E402
from diffik.solver import DiffIKSolver  # noqa: E402
from diffik.trajectory import ReachTrajectory  # noqa: E402

DEFAULT_SCENE = REPO_ROOT / "scene" / "panda_ik.xml"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "damping_sweep.csv"

# The six decade values are the required sweep. The extra points between 1e-4
# and 1e-2 resolve a band where the arm comes out of the singularity in a
# different posture and never recovers; without them that band shows up as a
# single anomalous decade rather than the wide, stable region it is. The
# ablation in results/article/ shows the clipping that follows is a symptom
# rather than the cause.
DAMPING_VALUES: tuple[float, ...] = (
    1e-6,
    1e-5,
    1e-4,
    3e-4,
    5e-4,
    1e-3,
    3e-3,
    5e-3,
    7e-3,
    1e-2,
    1e-1,
)

FIELDNAMES = (
    "damping",
    "step",
    "time",
    "position_error",
    "orientation_error",
    "max_dq",
    "condition_number",
    "clipped",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "seeds the global RNG. The trajectory is analytic and the run "
            "contains no randomness, so results do not depend on it today; the "
            "flag exists so a future stochastic variant stays reproducible "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--nullspace-gain",
        type=float,
        default=0.0,
        help=(
            "held at 0 for the sweep so the numbers describe the damping alone "
            "(default: %(default)s)"
        ),
    )
    return parser.parse_args(argv)


def run_one(
    handles: robot_model.RobotHandles,
    trajectory: ReachTrajectory,
    config: DiffIKConfig,
) -> list[dict[str, float]]:
    """Drive the trajectory once and record every step."""
    model, data = handles.model, handles.data
    solver = DiffIKSolver(handles, config)

    robot_model.reset_to_home(handles)
    robot_model.sync_target_to_site(handles)

    rows: list[dict[str, float]] = []
    for step in range(trajectory.steps(model.opt.timestep)):
        trajectory.apply(handles, data.time)
        mujoco.mj_forward(model, data)

        # Read the Jacobian before stepping: this is the matrix the solver is
        # about to use, at the configuration it is about to use it in.
        condition_number = float(np.linalg.cond(solver.jacobian()))

        solver.step()
        error = solver.pose_error.value

        rows.append(
            {
                "damping": config.damping,
                "step": step,
                "time": float(data.time),
                "position_error": float(np.linalg.norm(error[:3])),
                "orientation_error": float(np.linalg.norm(error[3:])),
                "max_dq": float(np.abs(solver.joint_velocity).max()),
                "condition_number": condition_number,
                "clipped": int(solver.clipped_last_step),
            }
        )

        mujoco.mj_step(model, data)

    return rows


def _trim(row: dict[str, float]) -> dict[str, float]:
    """Round floats to six significant digits before writing.

    Full double repr triples the file size with digits that carry no
    information: the quantities here are the result of thousands of physics
    steps, and nothing downstream reads past the sixth digit. Summaries are
    computed from the in-memory values, so the rounding only affects the file.
    """
    return {
        key: float(f"{value:.6g}") if isinstance(value, float) else value
        for key, value in row.items()
    }


def summarise(rows: list[dict[str, float]]) -> dict[str, float]:
    position = np.array([row["position_error"] for row in rows])
    orientation = np.array([row["orientation_error"] for row in rows])
    max_dq = np.array([row["max_dq"] for row in rows])
    clipped = np.array([row["clipped"] for row in rows])

    return {
        "damping": rows[0]["damping"],
        "rms_position_error": float(np.sqrt(np.mean(position**2))),
        "rms_orientation_error": float(np.sqrt(np.mean(orientation**2))),
        "peak_dq": float(max_dq.max()),
        "clipped_steps": int(clipped.sum()),
        "peak_condition_number": float(
            max(row["condition_number"] for row in rows)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    np.random.seed(args.seed)

    handles = robot_model.load(args.scene)
    trajectory = ReachTrajectory.from_home(handles)

    print(
        f"trajectory: radius {trajectory.start_radius:.3f} m -> "
        f"{trajectory.peak_radius:.3f} m -> {trajectory.start_radius:.3f} m "
        f"over {trajectory.duration:g} s "
        f"({trajectory.steps(handles.model.opt.timestep)} steps)"
    )

    all_rows: list[dict[str, float]] = []
    summaries: list[dict[str, float]] = []
    for damping in DAMPING_VALUES:
        config = DiffIKConfig(
            damping=damping, nullspace_gain=args.nullspace_gain
        )
        rows = run_one(handles, trajectory, config)
        all_rows.extend(rows)
        summaries.append(summarise(rows))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(_trim(row) for row in all_rows)

    print(f"\nwrote {len(all_rows)} rows to {args.output}\n")
    header = (
        f"{'damping':>9}  {'rms pos [m]':>12}  {'rms ori [rad]':>14}  "
        f"{'peak |dq|':>10}  {'clipped':>8}  {'peak cond(J)':>13}"
    )
    print(header)
    print("-" * len(header))
    for summary in summaries:
        print(
            f"{summary['damping']:>9.0e}  "
            f"{summary['rms_position_error']:>12.5f}  "
            f"{summary['rms_orientation_error']:>14.5f}  "
            f"{summary['peak_dq']:>10.2f}  "
            f"{summary['clipped_steps']:>8d}  "
            f"{summary['peak_condition_number']:>13.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
