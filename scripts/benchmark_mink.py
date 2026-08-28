"""Compare the damped least squares solver against mink's QP formulation.

Both methods drive the same trajectory from diffik.trajectory, for the same
duration, at the same timestep, and are measured with the same columns as
scripts/benchmark_damping.py.

The difference under test is how each one respects the joint limits. The DLS
implementation solves an unconstrained problem and clips the integrated
configuration afterwards, which changes the direction of the commanded motion.
mink states the joint and velocity limits as constraints of a quadratic
program, so the solution it returns is the best feasible direction rather than a
truncated infeasible one.

    python scripts/benchmark_mink.py

Writes results/mink_comparison.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import mink
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from diffik import model as robot_model  # noqa: E402
from diffik.config import DiffIKConfig  # noqa: E402
from diffik.solver import DiffIKSolver  # noqa: E402
from diffik.trajectory import ReachTrajectory  # noqa: E402

DEFAULT_SCENE = REPO_ROOT / "scene" / "panda_ik.xml"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "mink_comparison.csv"

# One under-damped value, one inside the band where the sweep showed the arm
# jamming against a limit, and one well damped.
DLS_DAMPING_VALUES: tuple[float, ...] = (1e-4, 1e-3, 1e-2)

# Shared velocity bound, in rad/s. The DLS runs scale dq down to reach it after
# the fact; mink receives it as a constraint. Using one number for both is what
# makes the comparison about enforcement rather than about limits.
VELOCITY_LIMIT = 1.0

# QP solver from qpsolvers. daqp ships as a mink dependency.
QP_SOLVER = "daqp"

FIELDNAMES = (
    "method",
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
            "seeds the global RNG. Both methods are deterministic and the "
            "trajectory is analytic, so results do not depend on it today "
            "(default: %(default)s)"
        ),
    )
    return parser.parse_args(argv)


def _trim(row: dict[str, object]) -> dict[str, object]:
    """Six significant digits; the rest is noise and doubles the file size."""
    return {
        key: float(f"{value:.6g}") if isinstance(value, float) else value
        for key, value in row.items()
    }


def run_dls(
    handles: robot_model.RobotHandles,
    trajectory: ReachTrajectory,
    damping: float,
    max_angvel: float,
) -> list[dict[str, object]]:
    model, data = handles.model, handles.data
    solver = DiffIKSolver(
        handles, DiffIKConfig(damping=damping, max_angvel=max_angvel)
    )

    robot_model.reset_to_home(handles)
    robot_model.sync_target_to_site(handles)

    rows: list[dict[str, object]] = []
    for step in range(trajectory.steps(model.opt.timestep)):
        trajectory.apply(handles, data.time)
        mujoco.mj_forward(model, data)
        condition_number = float(np.linalg.cond(solver.jacobian()))

        solver.step()
        error = solver.pose_error.value

        rows.append(
            {
                "method": "dls",
                "damping": damping,
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


def run_mink(
    handles: robot_model.RobotHandles, trajectory: ReachTrajectory
) -> list[dict[str, object]]:
    """Same loop, same measurements, with the QP in place of the DLS solve.

    The integration horizon handed to mink is the integration_dt the DLS side
    uses as its gain. That matters for more than step size: mink turns the
    configuration limits into velocity bounds over exactly that horizon, so a
    different value would compare two different problems.
    """
    model, data = handles.model, handles.data
    integration_dt = DiffIKConfig.integration_dt

    # Used only to report cond(J), to measure the error, and to clip-check on
    # the same footing as the DLS runs. Its solve() is never called.
    probe = DiffIKSolver(handles, DiffIKConfig())

    robot_model.reset_to_home(handles)
    robot_model.sync_target_to_site(handles)

    configuration = mink.Configuration(model, q=data.qpos.copy())

    frame_task = mink.FrameTask(
        frame_name=robot_model.SITE_NAME,
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    # Keeps the redundant dof from wandering, the job the nullspace term does on
    # the DLS side. Cheap relative to the frame task, so it never competes with
    # tracking.
    posture_task = mink.PostureTask(model, cost=1e-2)
    posture_task.set_target(model.key_qpos[handles.key_id].copy())
    tasks = [frame_task, posture_task]

    limits = [
        mink.ConfigurationLimit(model),
        mink.VelocityLimit(
            model,
            {name: VELOCITY_LIMIT for name in robot_model.ARM_JOINT_NAMES},
        ),
    ]

    rows: list[dict[str, object]] = []
    for step in range(trajectory.steps(model.opt.timestep)):
        trajectory.apply(handles, data.time)
        mujoco.mj_forward(model, data)
        condition_number = float(np.linalg.cond(probe.jacobian()))

        # Follow the physics, exactly as the DLS solver does by reading qpos.
        configuration.update(data.qpos.copy())
        frame_task.set_target(mink.SE3.from_mocap_name(model, data, "target"))

        velocity = mink.solve_ik(
            configuration, tasks, integration_dt, QP_SOLVER, limits=limits
        )
        configuration.integrate_inplace(velocity, integration_dt)

        command = configuration.q[handles.qpos_ids]
        clipped_command = np.clip(command, probe.lower_limits, probe.upper_limits)
        data.ctrl[handles.act_ids] = clipped_command

        error = probe.pose_error.compute()
        rows.append(
            {
                "method": "mink",
                "damping": "",
                "step": step,
                "time": float(data.time),
                "position_error": float(np.linalg.norm(error[:3])),
                "orientation_error": float(np.linalg.norm(error[3:])),
                "max_dq": float(np.abs(velocity[handles.dof_ids]).max()),
                "condition_number": condition_number,
                # Whether the solution needed truncating at all. A correct
                # constrained solve should never produce one.
                "clipped": int(bool(np.any(clipped_command != command))),
            }
        )
        mujoco.mj_step(model, data)

    return rows


def summarise(rows: list[dict[str, object]]) -> dict[str, object]:
    position = np.array([row["position_error"] for row in rows])
    orientation = np.array([row["orientation_error"] for row in rows])
    max_dq = np.array([row["max_dq"] for row in rows])
    clipped = np.array([row["clipped"] for row in rows])

    label = str(rows[0]["method"])
    if rows[0]["damping"] != "":
        label = "dls {:.0e}".format(rows[0]["damping"])

    return {
        "label": label,
        "rms_position_error": float(np.sqrt(np.mean(position**2))),
        "rms_orientation_error": float(np.sqrt(np.mean(orientation**2))),
        "peak_dq": float(max_dq.max()),
        "clipped_steps": int(clipped.sum()),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    np.random.seed(args.seed)

    handles = robot_model.load(args.scene)
    trajectory = ReachTrajectory.from_home(handles)

    print(
        f"trajectory: radius {trajectory.start_radius:.3f} m -> "
        f"{trajectory.peak_radius:.3f} m -> {trajectory.start_radius:.3f} m "
        f"over {trajectory.duration:g} s, velocity limit {VELOCITY_LIMIT:g} rad/s"
    )

    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for damping in DLS_DAMPING_VALUES:
        rows = run_dls(handles, trajectory, damping, VELOCITY_LIMIT)
        all_rows.extend(rows)
        summaries.append(summarise(rows))

    rows = run_mink(handles, trajectory)
    all_rows.extend(rows)
    summaries.append(summarise(rows))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(_trim(row) for row in all_rows)

    print(f"\nwrote {len(all_rows)} rows to {args.output}\n")
    header = (
        f"{'method':>11}  {'rms pos [m]':>12}  {'rms ori [rad]':>14}  "
        f"{'peak |dq|':>10}  {'clipped':>8}"
    )
    print(header)
    print("-" * len(header))
    for summary in summaries:
        print(
            f"{summary['label']:>11}  "
            f"{summary['rms_position_error']:>12.5f}  "
            f"{summary['rms_orientation_error']:>14.5f}  "
            f"{summary['peak_dq']:>10.2f}  "
            f"{summary['clipped_steps']:>8d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
