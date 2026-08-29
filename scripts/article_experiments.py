"""Supporting experiments for the write-up on the damping failure band.

Five experiments, all on the trajectory from diffik.trajectory, all writing to
results/article/. Nothing here touches the files already in results/.

    python scripts/article_experiments.py

E1  clip forensics: every clip event, and which joint goes first
E2  ablation at damping 1e-3: what removes the failure and what does not
E3  configuration traces for four runs
E4  where two runs first part company, relative to the first clip
E5  a 2D sweep over damping and nullspace gain

A note on the no-clip conditions. The Panda position actuators carry a
ctrlrange identical to the joint range, and MuJoCo clamps data.ctrl against it
by default, so switching off the solver's own clipping alone changes nothing
measurable. Those runs therefore also disable mjDSBL_CLAMPCTRL. Even then the
arm cannot leave its joint range: the limit constraints in the physics still
hold it. What the ablation isolates is whether redirecting the *setpoint* along
a limit surface is what causes the failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from diffik import model as robot_model  # noqa: E402
from diffik.config import DiffIKConfig  # noqa: E402
from diffik.model import ARM_JOINT_NAMES, RobotHandles  # noqa: E402
from diffik.solver import DiffIKSolver  # noqa: E402
from diffik.trajectory import ReachTrajectory  # noqa: E402

DEFAULT_SCENE = REPO_ROOT / "scene" / "panda_ik.xml"
OUTPUT_DIR = REPO_ROOT / "results" / "article"

# Any configuration this far from home, or any non-finite value, means the run
# has left the region where the numbers mean anything.
DIVERGENCE_THRESHOLD = 1e3

# E4: how far apart two joint configurations have to be to count as separated.
SEPARATION_THRESHOLD = 0.01


@dataclass
class ClipEvent:
    step: int
    time: float
    joint_name: str
    limit_side: str
    overshoot_rad: float


@dataclass
class RunResult:
    """Per-step record of one run, plus the derived run-level numbers."""

    time: NDArray[np.float64]
    position_error: NDArray[np.float64]
    orientation_error: NDArray[np.float64]
    max_dq: NDArray[np.float64]
    condition_number: NDArray[np.float64]
    distance_from_home: NDArray[np.float64]
    any_clipped: NDArray[np.int_]
    qpos: NDArray[np.float64]
    clip_events: list[ClipEvent] = field(default_factory=list)
    diverged: bool = False
    diverged_at_step: int | None = None

    @property
    def first_clip_step(self) -> int | None:
        return self.clip_events[0].step if self.clip_events else None

    @property
    def first_clip_joint(self) -> str | None:
        return self.clip_events[0].joint_name if self.clip_events else None

    def summary(self) -> dict[str, object]:
        rms = lambda values: float(np.sqrt(np.mean(values**2)))  # noqa: E731
        return {
            "rms_position_error": rms(self.position_error),
            "rms_orientation_error": rms(self.orientation_error),
            "final_position_error": float(self.position_error[-1]),
            "peak_dq": float(self.max_dq.max()),
            "clipped_steps": int(self.any_clipped.sum()),
            "peak_condition_number": float(self.condition_number.max()),
            "max_distance_from_home": float(self.distance_from_home.max()),
            "diverged": int(self.diverged),
        }


def run_once(
    handles: RobotHandles,
    trajectory: ReachTrajectory,
    config: DiffIKConfig,
    disable_ctrl_clamp: bool = False,
) -> RunResult:
    """Drive the trajectory once and record everything the experiments need.

    A diverging run stops early and is reported as such rather than being
    allowed to poison the rest of a sweep with NaN.
    """
    model, data = handles.model, handles.data
    solver = DiffIKSolver(handles, config)

    original_flags = int(model.opt.disableflags)
    if disable_ctrl_clamp:
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)

    try:
        robot_model.reset_to_home(handles)
        robot_model.sync_target_to_site(handles)

        steps = trajectory.steps(model.opt.timestep)
        result = RunResult(
            time=np.zeros(steps),
            position_error=np.zeros(steps),
            orientation_error=np.zeros(steps),
            max_dq=np.zeros(steps),
            condition_number=np.zeros(steps),
            distance_from_home=np.zeros(steps),
            any_clipped=np.zeros(steps, dtype=int),
            qpos=np.zeros((steps, 7)),
        )

        for step in range(steps):
            trajectory.apply(handles, data.time)
            mujoco.mj_forward(model, data)

            condition_number = float(np.linalg.cond(solver.jacobian()))
            solver.step()
            error = solver.pose_error.value
            arm = data.qpos[handles.qpos_ids]

            result.time[step] = data.time
            result.position_error[step] = np.linalg.norm(error[:3])
            result.orientation_error[step] = np.linalg.norm(error[3:])
            result.max_dq[step] = np.abs(solver.joint_velocity).max()
            result.condition_number[step] = condition_number
            result.distance_from_home[step] = np.linalg.norm(arm - handles.q_home)
            result.any_clipped[step] = int(solver.clipped_last_step)
            result.qpos[step] = arm

            for joint in np.flatnonzero(solver.clipped_joints):
                overshoot = float(solver.clip_overshoot[joint])
                result.clip_events.append(
                    ClipEvent(
                        step=step,
                        time=float(data.time),
                        joint_name=ARM_JOINT_NAMES[joint],
                        limit_side="upper" if overshoot > 0 else "lower",
                        overshoot_rad=overshoot,
                    )
                )

            mujoco.mj_step(model, data)

            if not np.all(np.isfinite(arm)) or np.abs(arm).max() > DIVERGENCE_THRESHOLD:
                result.diverged = True
                result.diverged_at_step = step
                # Freeze the remaining slots at the last valid sample so the
                # arrays stay a fixed length without inventing motion.
                for array in (
                    result.time,
                    result.position_error,
                    result.orientation_error,
                    result.max_dq,
                    result.condition_number,
                    result.distance_from_home,
                ):
                    array[step + 1 :] = array[step]
                result.any_clipped[step + 1 :] = result.any_clipped[step]
                result.qpos[step + 1 :] = result.qpos[step]
                break

        return result
    finally:
        model.opt.disableflags = original_flags


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: float(f"{value:.6g}") if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


# ---------------------------------------------------------------------------
# E1
# ---------------------------------------------------------------------------

E1_DAMPING = (3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 7e-3)


def experiment_1(handles: RobotHandles, trajectory: ReachTrajectory) -> dict:
    events: list[dict] = []
    first: list[dict] = []

    for damping in E1_DAMPING:
        result = run_once(handles, trajectory, DiffIKConfig(damping=damping))
        for event in result.clip_events:
            events.append(
                {
                    "damping": damping,
                    "step": event.step,
                    "time": event.time,
                    "joint_name": event.joint_name,
                    "limit_side": event.limit_side,
                    "overshoot_rad": event.overshoot_rad,
                }
            )
        joints = {event.joint_name for event in result.clip_events}
        first.append(
            {
                "damping": damping,
                "total_clip_events": len(result.clip_events),
                "clipped_steps": int(result.any_clipped.sum()),
                "first_clip_step": result.first_clip_step
                if result.first_clip_step is not None
                else "",
                "first_clip_time": result.clip_events[0].time
                if result.clip_events
                else "",
                "first_clip_joint": result.first_clip_joint or "",
                "first_clip_side": result.clip_events[0].limit_side
                if result.clip_events
                else "",
                "distinct_joints_clipped": " ".join(sorted(joints)),
            }
        )

    _write_csv(
        OUTPUT_DIR / "clip_events.csv",
        ("damping", "step", "time", "joint_name", "limit_side", "overshoot_rad"),
        events,
    )
    _write_csv(
        OUTPUT_DIR / "clip_first_events.csv",
        (
            "damping",
            "total_clip_events",
            "clipped_steps",
            "first_clip_step",
            "first_clip_time",
            "first_clip_joint",
            "first_clip_side",
            "distinct_joints_clipped",
        ),
        first,
    )
    return {"events": len(events), "runs": len(E1_DAMPING), "first": first}


# ---------------------------------------------------------------------------
# E2
# ---------------------------------------------------------------------------

E2_DAMPING = 1e-3
E2_CONDITIONS: tuple[tuple[str, dict, bool], ...] = (
    ("baseline", {}, False),
    ("velocity_clamp", {"max_angvel": 1.0}, False),
    ("no_clip", {"clip_joint_limits": False}, True),
    ("nullspace", {"nullspace_gain": 5.0}, False),
    (
        "nullspace_no_clip",
        {"nullspace_gain": 5.0, "clip_joint_limits": False},
        True,
    ),
)


def experiment_2(handles: RobotHandles, trajectory: ReachTrajectory) -> dict:
    rows = []
    for label, overrides, disable_clamp in E2_CONDITIONS:
        config = DiffIKConfig(damping=E2_DAMPING, **overrides)
        result = run_once(handles, trajectory, config, disable_ctrl_clamp=disable_clamp)
        rows.append({"label": label, **result.summary()})

    _write_csv(
        OUTPUT_DIR / "ablation.csv",
        (
            "label",
            "rms_position_error",
            "rms_orientation_error",
            "final_position_error",
            "peak_dq",
            "clipped_steps",
            "peak_condition_number",
            "max_distance_from_home",
            "diverged",
        ),
        rows,
    )
    return {"rows": rows}


# ---------------------------------------------------------------------------
# E3
# ---------------------------------------------------------------------------

# The fourth trace is the control the other three cannot provide: a healthy run
# at the same damping as the failing one, so the divergence figure is not just
# comparing two damping values.
E3_RUNS: tuple[tuple[float, float], ...] = (
    (3e-4, 0.0),
    (1e-3, 0.0),
    (1e-2, 0.0),
    (1e-3, 5.0),
)


def experiment_3(
    handles: RobotHandles, trajectory: ReachTrajectory
) -> dict[tuple[float, float], RunResult]:
    rows: list[dict] = []
    traces: dict[tuple[float, float], RunResult] = {}

    for damping, gain in E3_RUNS:
        result = run_once(
            handles,
            trajectory,
            DiffIKConfig(damping=damping, nullspace_gain=gain),
        )
        traces[(damping, gain)] = result
        for step in range(len(result.time)):
            row = {
                "damping": damping,
                "nullspace_gain": gain,
                "step": step,
                "time": float(result.time[step]),
            }
            for joint in range(7):
                row[f"joint{joint + 1}"] = float(result.qpos[step, joint])
            row["position_error"] = float(result.position_error[step])
            row["condition_number"] = float(result.condition_number[step])
            row["distance_from_home"] = float(result.distance_from_home[step])
            row["any_clipped"] = int(result.any_clipped[step])
            rows.append(row)

    _write_csv(
        OUTPUT_DIR / "qpos_traces.csv",
        (
            "damping",
            "nullspace_gain",
            "step",
            "time",
            *(f"joint{i + 1}" for i in range(7)),
            "position_error",
            "condition_number",
            "distance_from_home",
            "any_clipped",
        ),
        rows,
    )
    return traces


# ---------------------------------------------------------------------------
# E4
# ---------------------------------------------------------------------------


def experiment_4(
    traces: dict[tuple[float, float], RunResult], trajectory: ReachTrajectory
) -> dict:
    failing = traces[(1e-3, 0.0)]
    healthy = traces[(1e-2, 0.0)]

    separation = np.linalg.norm(failing.qpos - healthy.qpos, axis=1)
    crossed = np.flatnonzero(separation > SEPARATION_THRESHOLD)
    if crossed.size == 0:
        payload = {"separated": False, "threshold_rad": SEPARATION_THRESHOLD}
        (OUTPUT_DIR / "divergence.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        return payload

    step = int(crossed[0])
    per_joint = np.abs(failing.qpos[step] - healthy.qpos[step])
    dominant = int(np.argmax(per_joint))
    first_clip = failing.first_clip_step

    payload = {
        "threshold_rad": SEPARATION_THRESHOLD,
        "separated": True,
        "divergence_step": step,
        "divergence_time": float(failing.time[step]),
        "separation_at_divergence_rad": float(separation[step]),
        "dominant_joint": ARM_JOINT_NAMES[dominant],
        "dominant_joint_difference_rad": float(per_joint[dominant]),
        "per_joint_difference_rad": {
            ARM_JOINT_NAMES[i]: float(per_joint[i]) for i in range(7)
        },
        "condition_number": {
            "damping_1e-3": float(failing.condition_number[step]),
            "damping_1e-2": float(healthy.condition_number[step]),
        },
        "distance_from_home_rad": {
            "damping_1e-3": float(failing.distance_from_home[step]),
            "damping_1e-2": float(healthy.distance_from_home[step]),
        },
        "target_radius_m": float(
            np.linalg.norm(
                trajectory.position_at(float(failing.time[step]))
                - np.array([0.0, 0.0, 0.333])
            )
        ),
        "first_clip_step_damping_1e-3": first_clip,
        "first_clip_time_damping_1e-3": (
            float(failing.time[first_clip]) if first_clip is not None else None
        ),
        "first_clip_joint_damping_1e-3": failing.first_clip_joint,
        "divergence_precedes_first_clip": (
            None if first_clip is None else bool(step < first_clip)
        ),
        "steps_between_divergence_and_first_clip": (
            None if first_clip is None else int(first_clip - step)
        ),
    }
    (OUTPUT_DIR / "divergence.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


# ---------------------------------------------------------------------------
# E5
# ---------------------------------------------------------------------------

E5_DAMPING = (1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2)
E5_GAINS = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)


def experiment_5(handles: RobotHandles, trajectory: ReachTrajectory) -> dict:
    rows = []
    for damping in E5_DAMPING:
        for gain in E5_GAINS:
            config = DiffIKConfig(damping=damping, nullspace_gain=gain)
            try:
                result = run_once(handles, trajectory, config)
                summary = result.summary()
            except Exception as error:  # noqa: BLE001
                # A run that blows up is data, not a reason to lose the other 41.
                summary = {
                    "rms_position_error": float("nan"),
                    "rms_orientation_error": float("nan"),
                    "final_position_error": float("nan"),
                    "peak_dq": float("nan"),
                    "clipped_steps": 0,
                    "peak_condition_number": float("nan"),
                    "max_distance_from_home": float("nan"),
                    "diverged": 1,
                }
                print(f"  damping {damping:.0e} Kn {gain}: raised {error!r}")
            rows.append({"damping": damping, "nullspace_gain": gain, **summary})

    _write_csv(
        OUTPUT_DIR / "sweep_2d.csv",
        (
            "damping",
            "nullspace_gain",
            "rms_position_error",
            "rms_orientation_error",
            "final_position_error",
            "peak_dq",
            "clipped_steps",
            "peak_condition_number",
            "max_distance_from_home",
            "diverged",
        ),
        rows,
    )
    return {"rows": rows}


# ---------------------------------------------------------------------------
# E6
# ---------------------------------------------------------------------------

# The whole study so far rests on one trajectory. If the band only exists at
# that exact reach, the finding is a coincidence of the path rather than a
# property of the solver, so the same grid is repeated either side of it.
E6_RADII = (0.66, 0.68, 0.69, 0.70, 0.71, 0.72)
E6_GAINS = (0.0, 2.0, 5.0)


def experiment_6(handles: RobotHandles) -> dict:
    rows = []
    for radius in E6_RADII:
        trajectory = ReachTrajectory.from_home(handles, peak_radius=radius)
        for damping in E5_DAMPING:
            for gain in E6_GAINS:
                result = run_once(
                    handles,
                    trajectory,
                    DiffIKConfig(damping=damping, nullspace_gain=gain),
                )
                row = {
                    "peak_radius": radius,
                    "damping": damping,
                    "nullspace_gain": gain,
                    **result.summary(),
                }
                # The final posture, not just its distance from home. Two
                # different configurations can share a norm; only the angles
                # themselves can show whether the failing runs land in one
                # place.
                for joint in range(7):
                    row[f"final_joint{joint + 1}"] = float(result.qpos[-1, joint])
                rows.append(row)

    _write_csv(
        OUTPUT_DIR / "radius_robustness.csv",
        (
            "peak_radius",
            "damping",
            "nullspace_gain",
            "rms_position_error",
            "rms_orientation_error",
            "final_position_error",
            "peak_dq",
            "clipped_steps",
            "peak_condition_number",
            "max_distance_from_home",
            "diverged",
            *(f"final_joint{i + 1}" for i in range(7)),
        ),
        rows,
    )
    return {"rows": rows}


# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    np.random.seed(args.seed)

    handles = robot_model.load(args.scene)
    trajectory = ReachTrajectory.from_home(handles)
    steps = trajectory.steps(handles.model.opt.timestep)

    print(
        f"trajectory {trajectory.start_radius:.3f} -> {trajectory.peak_radius:.3f} m, "
        f"{steps} steps at {handles.model.opt.timestep} s"
    )

    timings: dict[str, float] = {}

    started = time.perf_counter()
    e1 = experiment_1(handles, trajectory)
    timings["E1_clip_forensics"] = time.perf_counter() - started
    print(f"E1  {e1['events']} clip events over {e1['runs']} runs")
    for row in e1["first"]:
        print(
            f"    damping {row['damping']:.0e}  first clip "
            f"{row['first_clip_joint'] or 'none':>8}  "
            f"step {row['first_clip_step']}  joints [{row['distinct_joints_clipped']}]"
        )

    started = time.perf_counter()
    e2 = experiment_2(handles, trajectory)
    timings["E2_ablation"] = time.perf_counter() - started
    print("E2  ablation at damping 1e-3")
    for row in e2["rows"]:
        print(
            f"    {row['label']:>18}  rms {row['rms_position_error']:.5f}  "
            f"final {row['final_position_error']:.5f}  "
            f"clips {row['clipped_steps']:>4}  "
            f"drift {row['max_distance_from_home']:.3f}"
        )

    started = time.perf_counter()
    traces = experiment_3(handles, trajectory)
    timings["E3_traces"] = time.perf_counter() - started
    print(f"E3  {len(traces)} traces written")

    started = time.perf_counter()
    e4 = experiment_4(traces, trajectory)
    timings["E4_divergence"] = time.perf_counter() - started
    if e4.get("separated"):
        print(
            f"E4  diverges at step {e4['divergence_step']} "
            f"(t={e4['divergence_time']:.3f}s), first clip at step "
            f"{e4['first_clip_step_damping_1e-3']}, "
            f"divergence first: {e4['divergence_precedes_first_clip']}"
        )
    else:
        print("E4  the two runs never separated")

    started = time.perf_counter()
    e5 = experiment_5(handles, trajectory)
    timings["E5_sweep_2d"] = time.perf_counter() - started
    diverged = sum(int(row["diverged"]) for row in e5["rows"])
    print(f"E5  {len(e5['rows'])} runs, {diverged} diverged")

    started = time.perf_counter()
    e6 = experiment_6(handles)
    timings["E6_radius_robustness"] = time.perf_counter() - started
    failing = [row for row in e6["rows"] if row["rms_position_error"] > 0.05]
    print(f"E6  {len(e6['rows'])} runs, {len(failing)} failing")
    for radius in E6_RADII:
        at_radius = [
            row
            for row in failing
            if row["peak_radius"] == radius and row["nullspace_gain"] == 0.0
        ]
        band = ", ".join(f"{row['damping']:.0e}" for row in at_radius)
        print(f"    radius {radius:.2f}  gain 0 fails at [{band or 'none'}]")

    timings["total"] = sum(timings.values())
    (OUTPUT_DIR / "timings.json").write_text(
        json.dumps(timings, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\ntotal {timings['total']:.1f} s, written to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
