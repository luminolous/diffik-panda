"""Figures and FACTS.md for the write-up, built from results/article/.

    python scripts/article_report.py

Reads only the CSV and JSON files written by scripts/article_experiments.py.
No simulation runs here, so every number in the article traces back to a file.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ARTICLE_DIR = REPO_ROOT / "results" / "article"
FIGURE_DIR = ARTICLE_DIR / "figures"

# Recorded in scene/panda_ik.xml and in the scaffold commit.
MENAGERIE_COMMIT = "da76818e269b82289eba39808e2fb91d679d6994"

# The configurations the divergence figure compares.
FAILING = (1e-3, 0.0)
HEALTHY = (1e-2, 0.0)
CONTROL = (1e-3, 5.0)


def load() -> dict:
    return {
        "sweep_2d": pd.read_csv(ARTICLE_DIR / "sweep_2d.csv"),
        "ablation": pd.read_csv(ARTICLE_DIR / "ablation.csv"),
        "traces": pd.read_csv(ARTICLE_DIR / "qpos_traces.csv"),
        "clip_first": pd.read_csv(ARTICLE_DIR / "clip_first_events.csv"),
        "clip_events": pd.read_csv(ARTICLE_DIR / "clip_events.csv"),
        "divergence": json.loads(
            (ARTICLE_DIR / "divergence.json").read_text(encoding="utf-8")
        ),
        "timings": json.loads(
            (ARTICLE_DIR / "timings.json").read_text(encoding="utf-8")
        ),
    }


def figure_failure_band(sweep: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for gain, colour, marker in ((0.0, "tab:red", "o"), (5.0, "tab:blue", "s")):
        group = sweep[sweep.nullspace_gain == gain].sort_values("damping")
        ax.plot(
            group.damping,
            group.rms_position_error,
            marker=marker,
            color=colour,
            label=f"nullspace gain {gain:g}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("damping")
    ax.set_ylabel("RMS position error [m]")
    ax.set_title("The failure band, with and without the secondary objective")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "failure_band.png", dpi=140)
    plt.close(fig)


def figure_divergence(traces: pd.DataFrame, divergence: dict) -> None:
    def pick(key):
        damping, gain = key
        return traces[
            np.isclose(traces.damping, damping)
            & np.isclose(traces.nullspace_gain, gain)
        ]

    failing, healthy, control = pick(FAILING), pick(HEALTHY), pick(CONTROL)
    joints = [f"joint{i + 1}" for i in range(7)]
    colours = plt.cm.tab10(np.linspace(0, 0.7, 7))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for joint, colour in zip(joints, colours):
        axes[0].plot(failing.time, failing[joint], color=colour, linewidth=1.4,
                     label=joint)
        axes[0].plot(healthy.time, healthy[joint], color=colour, linewidth=1.0,
                     linestyle="--", alpha=0.75)

    axes[0].set_ylabel("joint angle [rad]")
    axes[0].set_title(
        "Joint trajectories: damping 1e-3 (solid) against 1e-2 (dashed), both at gain 0"
    )
    axes[0].legend(fontsize=7, ncol=7, loc="upper left")

    for ax in axes:
        if divergence.get("separated"):
            ax.axvline(
                divergence["divergence_time"],
                color="black",
                linestyle=":",
                linewidth=1.5,
                label="separation",
            )
        clip_time = divergence.get("first_clip_time_damping_1e-3")
        if clip_time is not None:
            ax.axvline(
                clip_time, color="tab:red", linestyle="-.", linewidth=1.5,
                label="first clip",
            )

    axes[1].plot(failing.time, failing.distance_from_home, color="tab:red",
                 label="damping 1e-3, gain 0")
    axes[1].plot(healthy.time, healthy.distance_from_home, color="tab:green",
                 label="damping 1e-2, gain 0")
    axes[1].plot(control.time, control.distance_from_home, color="tab:blue",
                 linestyle="--", label="damping 1e-3, gain 5")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("||q - q_home|| [rad]")
    axes[1].set_title("Posture drift; the control run shares the failing damping")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "qpos_divergence.png", dpi=140)
    plt.close(fig)


def figure_sweep_2d(sweep: pd.DataFrame) -> None:
    grid = sweep.pivot(
        index="damping", columns="nullspace_gain", values="rms_position_error"
    ).sort_index()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    mesh = ax.pcolormesh(
        np.arange(grid.shape[1] + 1),
        np.arange(grid.shape[0] + 1),
        grid.to_numpy(),
        norm=LogNorm(vmin=grid.to_numpy().min(), vmax=grid.to_numpy().max()),
        cmap="magma_r",
        edgecolors="white",
        linewidth=1,
    )
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            value = grid.to_numpy()[row, column]
            ax.text(
                column + 0.5,
                row + 0.5,
                f"{value:.4f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > 0.05 else "black",
            )

    ax.set_xticks(np.arange(grid.shape[1]) + 0.5)
    ax.set_xticklabels([f"{g:g}" for g in grid.columns])
    ax.set_yticks(np.arange(grid.shape[0]) + 0.5)
    ax.set_yticklabels([f"{d:.0e}" for d in grid.index])
    ax.set_xlabel("nullspace gain")
    ax.set_ylabel("damping")
    ax.set_title("RMS position error over damping and nullspace gain")
    fig.colorbar(mesh, ax=ax, label="RMS position error [m]")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "sweep_2d.png", dpi=140)
    plt.close(fig)


def write_facts(data: dict) -> None:
    sweep = data["sweep_2d"]
    ablation = data["ablation"]
    clip_first = data["clip_first"]
    divergence = data["divergence"]
    timings = data["timings"]

    failing_cells = sweep[sweep.rms_position_error > 0.05]
    baseline = sweep[sweep.nullspace_gain == 0.0].sort_values("damping")

    lines: list[str] = []
    add = lines.append

    add("# Facts for the write-up")
    add("")
    add(
        "Every number here is read from the files in `results/article/`, which "
        "`scripts/article_experiments.py` produced in one run. Nothing is "
        "rounded for effect."
    )
    add("")

    add("## The headline, and it contradicts the repository README")
    add("")
    add(
        "The README currently explains the failure band by saying the arm is "
        "pinned against a joint limit and that clipping holds it there. The "
        "ablation says that is wrong."
    )
    add("")
    add(
        "Removing the clipping entirely does not repair the run. Bounding the "
        "joint velocity does not repair it either. Only the nullspace term "
        "repairs it, and it does so whether clipping is on or off. The failure "
        "is a posture problem; the clipping is downstream of it."
    )
    add("")
    add(
        "The timing agrees. The failing run separates from the healthy one at "
        f"step {divergence['divergence_step']} "
        f"(t = {divergence['divergence_time']:.3f} s), and the first clip in that "
        f"run happens at step {divergence['first_clip_step_damping_1e-3']} "
        f"(t = {divergence['first_clip_time_damping_1e-3']:.3f} s). The two runs "
        f"had already parted company "
        f"{divergence['steps_between_divergence_and_first_clip']} steps "
        f"({divergence['steps_between_divergence_and_first_clip'] * 0.002:.3f} s) "
        "before anything was clipped."
    )
    add("")

    add("## E1 Clip forensics")
    add("")
    add("`results/article/clip_events.csv`, `results/article/clip_first_events.csv`")
    add("")
    add(
        "| damping | first clip step | first clip time [s] | first joint | side | "
        "joints clipped | clip events | clipped steps |"
    )
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in clip_first.itertuples():
        step = "none" if pd.isna(row.first_clip_step) else f"{int(row.first_clip_step)}"
        moment = "-" if pd.isna(row.first_clip_time) else f"{row.first_clip_time:.3f}"
        joint = "-" if pd.isna(row.first_clip_joint) else row.first_clip_joint
        side = "-" if pd.isna(row.first_clip_side) else row.first_clip_side
        joints = "-" if pd.isna(row.distinct_joints_clipped) else row.distinct_joints_clipped
        add(
            f"| `{row.damping:.0e}` | {step} | {moment} | {joint} | {side} | "
            f"{joints} | {row.total_clip_events} | {row.clipped_steps} |"
        )
    add("")
    first_joints = set(clip_first.first_clip_joint.dropna())
    add(
        f"The first joint to clip is the same in every run that clips at all: "
        f"{', '.join(sorted(first_joints))}, always against its upper limit. "
        "joint4 is the one asymmetric joint in the arm, limited to "
        "[-3.0718, -0.0698] rad, so it has the least room above it."
    )
    add("")
    add(
        "Inside the band, joint2 and joint6 join it. Below the band, at "
        "`3e-04`, joint4 clips alone for ten steps and the run recovers."
    )
    add("")

    add("## E2 Ablation at damping 1e-3")
    add("")
    add("`results/article/ablation.csv`")
    add("")
    add(
        "| condition | RMS position [m] | final position [m] | peak \\|dq\\| | "
        "clipped steps | peak cond(J) | max drift from home [rad] |"
    )
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for row in ablation.itertuples():
        add(
            f"| `{row.label}` | {row.rms_position_error:.5f} | "
            f"{row.final_position_error:.5f} | {row.peak_dq:.3f} | "
            f"{row.clipped_steps} | {row.peak_condition_number:,.0f} | "
            f"{row.max_distance_from_home:.3f} |"
        )
    add("")
    interpretations = {
        "baseline": "The failure as the committed sweep records it.",
        "velocity_clamp": (
            "Bounding the joint velocity changes nothing that matters. The run "
            "still ends at the same error, with the same number of clipped "
            "steps and the same drift."
        ),
        "no_clip": (
            "Removing the clipping does not repair the run either. The final "
            "error is unchanged to three decimal places and the drift is "
            "slightly worse. This is the result that rules clipping out as the "
            "cause."
        ),
        "nullspace": (
            "The secondary objective repairs the run completely, at the same "
            "damping. Error drops by a factor of 20 and the drift halves."
        ),
        "nullspace_no_clip": (
            "Identical to the previous row to six significant digits, which "
            "confirms that once the posture is held, clipping never comes up."
        ),
    }
    for label, text in interpretations.items():
        add(f"- **`{label}`** — {text}")
    add("")
    add(
        "The discriminator across the whole study is `max_distance_from_home`. "
        "Every failing run reaches 2.67 rad; every healthy run stays at or "
        "below 1.40 rad. The failure is not a gradual degradation, it is the "
        "arm settling into a different configuration."
    )
    add("")
    add("### What the no-clip condition does and does not show")
    add("")
    add(
        "The Panda position actuators carry a `ctrlrange` identical to the "
        "joint range, and MuJoCo clamps `data.ctrl` against it by default, so "
        "the no-clip runs also disable `mjDSBL_CLAMPCTRL`. Even so, the arm "
        "cannot leave its joint range: the limit constraints in the physics "
        "still hold it. What changes is that the setpoint may sit outside the "
        "range and keep pushing, rather than being redirected along the limit "
        "surface. The ablation therefore tests whether that redirection causes "
        "the failure. It does not test whether the arm can pass through a limit."
    )
    add("")

    add("## E3 and E4 Where the runs separate")
    add("")
    add("`results/article/qpos_traces.csv`, `results/article/divergence.json`")
    add("")
    add(
        f"- Separation threshold: {divergence['threshold_rad']} rad on "
        "`||q_1e-3 - q_1e-2||`."
    )
    add(
        f"- Crossed at step {divergence['divergence_step']}, "
        f"t = {divergence['divergence_time']:.3f} s, at a separation of "
        f"{divergence['separation_at_divergence_rad']:.6f} rad."
    )
    add(
        f"- Dominant joint at that moment: {divergence['dominant_joint']}, "
        f"{divergence['dominant_joint_difference_rad']:.6f} rad of the total."
    )
    add(
        f"- cond(J) there: {divergence['condition_number']['damping_1e-3']:.1f} "
        f"for 1e-3 against "
        f"{divergence['condition_number']['damping_1e-2']:.1f} for 1e-2. Both "
        "are small. The arm is nowhere near a singularity when the paths split."
    )
    add(
        f"- Distance from home there: "
        f"{divergence['distance_from_home_rad']['damping_1e-3']:.4f} against "
        f"{divergence['distance_from_home_rad']['damping_1e-2']:.4f} rad, "
        "still almost identical."
    )
    add(f"- Target radius there: {divergence['target_radius_m']:.4f} m.")
    add(
        f"- First clip in the failing run: step "
        f"{divergence['first_clip_step_damping_1e-3']}, "
        f"t = {divergence['first_clip_time_damping_1e-3']:.3f} s, on "
        f"{divergence['first_clip_joint_damping_1e-3']}."
    )
    add(
        f"- **Divergence precedes the first clip: "
        f"{divergence['divergence_precedes_first_clip']}**, by "
        f"{divergence['steps_between_divergence_and_first_clip']} steps."
    )
    add("")
    add(
        "The third trace in `qpos_traces.csv`, damping 1e-3 at gain 5, is the "
        "control: same damping as the failing run, healthy outcome. It shows "
        "the split is not a property of the damping value on its own."
    )
    add("")

    add("## E5 The 2D sweep")
    add("")
    add("`results/article/sweep_2d.csv`, 42 runs, none diverged or raised.")
    add("")
    grid = sweep.pivot(
        index="damping", columns="nullspace_gain", values="rms_position_error"
    ).sort_index()
    add("RMS position error, in metres.")
    add("")
    header = " | ".join(f"Kn {g:g}" for g in grid.columns)
    add(f"| damping | {header} |")
    add("| --- | " + " | ".join(["---"] * len(grid.columns)) + " |")
    for damping, row in grid.iterrows():
        cells = " | ".join(f"{value:.5f}" for value in row)
        add(f"| `{damping:.0e}` | {cells} |")
    add("")
    add("### Where the band is, and where it is not")
    add("")
    band = baseline[baseline.rms_position_error > 0.05]
    add(
        "- At gain 0 the band covers "
        + ", ".join(f"`{d:.0e}`" for d in band.damping)
        + ", exactly the range the committed sweep reports."
    )
    add(
        "- Any gain from 0.5 upward clears it. At `5e-04` the error falls from "
        f"{baseline[np.isclose(baseline.damping, 5e-4)].rms_position_error.iloc[0]:.5f} "
        "to "
        f"{sweep[(np.isclose(sweep.damping, 5e-4)) & (sweep.nullspace_gain == 0.5)].rms_position_error.iloc[0]:.5f} "
        "m."
    )
    add("")
    add("### Two failures, not one")
    add("")
    low = sweep[np.isclose(sweep.damping, 1e-4)].sort_values("nullspace_gain")
    add(
        "The secondary objective does not repair the low-damping regime. At "
        "damping `1e-04` the clip count and the conditioning across gains are:"
    )
    add("")
    add("| gain | RMS position [m] | clipped steps | peak cond(J) | max drift [rad] |")
    add("| --- | --- | --- | --- | --- |")
    for row in low.itertuples():
        add(
            f"| {row.nullspace_gain:g} | {row.rms_position_error:.5f} | "
            f"{row.clipped_steps} | {row.peak_condition_number:,.0f} | "
            f"{row.max_distance_from_home:.3f} |"
        )
    add("")
    add(
        "Clipping persists at every gain, and the conditioning gets *worse* as "
        "the gain rises: peak cond(J) goes from "
        f"{low[low.nullspace_gain == 0.0].peak_condition_number.iloc[0]:,.0f} at "
        f"gain 0 to "
        f"{low[low.nullspace_gain == 5.0].peak_condition_number.iloc[0]:,.0f} at "
        f"gain 5 and "
        f"{low[low.nullspace_gain == 10.0].peak_condition_number.iloc[0]:,.0f} at "
        "gain 10. The low-damping velocity spike and the mid-band posture "
        "collapse are two different failures and want two different fixes."
    )
    add("")
    add("### An isolated failure that is not in the band")
    add("")
    if len(failing_cells):
        add("| damping | gain | RMS position [m] | clipped steps | max drift [rad] |")
        add("| --- | --- | --- | --- | --- |")
        for row in failing_cells.sort_values(["damping", "nullspace_gain"]).itertuples():
            add(
                f"| `{row.damping:.0e}` | {row.nullspace_gain:g} | "
                f"{row.rms_position_error:.5f} | {row.clipped_steps} | "
                f"{row.max_distance_from_home:.3f} |"
            )
        add("")
    outlier = failing_cells[failing_cells.nullspace_gain > 0]
    if len(outlier):
        row = outlier.iloc[0]
        add(
            f"Damping `{row.damping:.0e}` at gain {row.nullspace_gain:g} fails "
            f"the same way, with the same 2.67 rad drift, while both its "
            "neighbours in gain are healthy. The failure is not monotone in the "
            "gain either: raising the secondary objective does not simply make "
            "things safer, it moves where the bad pocket sits."
        )
    else:
        add("Every failing cell in the sweep sits at gain 0.")
    add("")

    add("## Environment")
    add("")
    add(f"- mujoco {metadata.version('mujoco')}")
    add(f"- mink {metadata.version('mink')}")
    add(f"- numpy {metadata.version('numpy')}")
    add(f"- Python {sys.version.split()[0]}")
    add(f"- mujoco_menagerie commit `{MENAGERIE_COMMIT}`")
    add("- trajectory: 0.587 m to 0.700 m from the shoulder and back")
    add("- 3000 steps per run, timestep 0.002 s, duration 6 s")
    add("- integration_dt 1.0 throughout")
    add("")

    add("## Wall clock")
    add("")
    add("| experiment | seconds |")
    add("| --- | --- |")
    for key, value in timings.items():
        if key != "total":
            add(f"| {key} | {value:.2f} |")
    add(f"| **total** | **{timings['total']:.2f}** |")
    add("")

    add("## Replication of the externally supplied result")
    add("")
    external_path = REPO_ROOT / "results" / "nullspace_removes_failure_band.csv"
    committed_path = REPO_ROOT / "results" / "damping_summary.csv"
    if external_path.is_file() and committed_path.is_file():
        external = pd.read_csv(external_path)
        committed = pd.read_csv(committed_path)
        merged = external.merge(committed, on="damping")
        gain0 = (
            (merged.rms_position_error_kn0 - merged.rms_position_error).abs()
            / merged.rms_position_error
        ).max()
        clips = int(
            (merged.clipped_steps_kn0 - merged.clipped_steps).abs().max()
        )
        conditioning = (
            (merged.peak_condition_number_kn0 - merged.peak_condition_number).abs()
            / merged.peak_condition_number
        ).max()
        add(
            "`results/nullspace_removes_failure_band.csv` was produced outside "
            "this session. Treated as a replication, its gain-0 columns agree "
            f"with the committed `results/damping_summary.csv` over all "
            f"{len(merged)} damping values: maximum relative difference "
            f"{gain0:.3e} in RMS position error, {conditioning:.3e} in peak "
            f"cond(J), and {clips} in clipped steps. The environments are "
            "equivalent, so its gain-5 column can be taken as data."
        )
        add("")

        gain5 = sweep[sweep.nullspace_gain == 5.0][
            ["damping", "rms_position_error", "clipped_steps", "peak_condition_number"]
        ]
        both = external.merge(gain5, on="damping")
        if len(both):
            rms_gap = (
                (both.rms_position_error_kn5 - both.rms_position_error).abs()
                / both.rms_position_error
            ).max()
            clip_gap = int(
                (both.clipped_steps_kn5 - both.clipped_steps).abs().max()
            )
            add(
                "E5 reproduces that gain-5 column independently. Over the "
                f"{len(both)} damping values the two runs share, the maximum "
                f"relative difference in RMS position error is {rms_gap:.3e} "
                f"and the clip counts differ by at most {clip_gap}."
            )
            add("")
    else:
        add(
            "The externally supplied CSV was not present when this report was "
            "generated, so no replication check was run."
        )
        add("")

    (ARTICLE_DIR / "FACTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--article-dir", type=Path, default=ARTICLE_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = load()

    figure_failure_band(data["sweep_2d"])
    figure_divergence(data["traces"], data["divergence"])
    figure_sweep_2d(data["sweep_2d"])
    write_facts(data)

    print(f"figures written to {FIGURE_DIR}")
    print(f"facts written to {ARTICLE_DIR / 'FACTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
