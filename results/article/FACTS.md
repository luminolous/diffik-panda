# Facts for the write-up

Every number here is read from the files in `results/article/`, which `scripts/article_experiments.py` produced in one run. Nothing is rounded for effect.

## The headline, and it overturned the earlier explanation

Before these experiments, this repository explained the failure band by saying the arm was pinned against a joint limit and that clipping held it there. That reading came from the correlation in the sweep: every failing run clips more than a thousand steps, every healthy one clips a dozen or none. The ablation shows the correlation runs the other way round. The README and the results notebook have since been corrected.

Removing the clipping entirely does not repair the run. Bounding the joint velocity does not repair it either. Only the nullspace term repairs it, and it does so whether clipping is on or off. The failure is a posture problem; the clipping is downstream of it.

The timing agrees. The failing run separates from the healthy one at step 1401 (t = 2.802 s), and the first clip in that run happens at step 1860 (t = 3.720 s). The two runs had already parted company 459 steps (0.918 s) before anything was clipped.

## E1 Clip forensics

`results/article/clip_events.csv`, `results/article/clip_first_events.csv`

| damping | first clip step | first clip time [s] | first joint | side | joints clipped | clip events | clipped steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `3e-04` | 1489 | 2.978 | joint4 | upper | joint4 | 10 | 10 |
| `5e-04` | 1489 | 2.978 | joint4 | upper | joint2 joint4 joint6 | 1774 | 1145 |
| `1e-03` | 1860 | 3.720 | joint4 | upper | joint2 joint4 joint6 | 1769 | 1140 |
| `3e-03` | 1860 | 3.720 | joint4 | upper | joint2 joint4 joint6 | 1765 | 1140 |
| `5e-03` | 1860 | 3.720 | joint4 | upper | joint2 joint4 joint6 | 1757 | 1140 |
| `7e-03` | none | - | - | - | - | 0 | 0 |

The first joint to clip is the same in every run that clips at all: joint4, always against its upper limit. joint4 is the one asymmetric joint in the arm, limited to [-3.0718, -0.0698] rad, so it has the least room above it.

Inside the band, joint2 and joint6 join it. Below the band, at `3e-04`, joint4 clips alone for ten steps and the run recovers.

## E2 Ablation at damping 1e-3

`results/article/ablation.csv`

| condition | RMS position [m] | final position [m] | peak \|dq\| | clipped steps | peak cond(J) | max drift from home [rad] |
| --- | --- | --- | --- | --- | --- | --- |
| `baseline` | 0.17572 | 0.43017 | 3.068 | 1140 | 693,352 | 2.673 |
| `velocity_clamp` | 0.17434 | 0.43009 | 1.000 | 1140 | 693,352 | 2.673 |
| `no_clip` | 0.17258 | 0.42990 | 3.090 | 0 | 693,352 | 2.699 |
| `nullspace` | 0.00888 | 0.00657 | 0.064 | 0 | 260 | 1.306 |
| `nullspace_no_clip` | 0.00888 | 0.00657 | 0.064 | 0 | 260 | 1.306 |

- **`baseline`** — The failure as the committed sweep records it.
- **`velocity_clamp`** — Bounding the joint velocity changes nothing that matters. The run still ends at the same error, with the same number of clipped steps and the same drift.
- **`no_clip`** — Removing the clipping does not repair the run either. The final error is unchanged to three decimal places and the drift is slightly worse. This is the result that rules clipping out as the cause.
- **`nullspace`** — The secondary objective repairs the run completely, at the same damping. Error drops by a factor of 20 and the drift halves.
- **`nullspace_no_clip`** — Identical to the previous row to six significant digits, which confirms that once the posture is held, clipping never comes up.

The discriminator across the whole study is `max_distance_from_home`. Every failing run reaches 2.67 rad; every healthy run stays at or below 1.40 rad. The failure is not a gradual degradation, it is the arm settling into a different configuration.

### What the no-clip condition does and does not show

The Panda position actuators carry a `ctrlrange` identical to the joint range, and MuJoCo clamps `data.ctrl` against it by default, so the no-clip runs also disable `mjDSBL_CLAMPCTRL`. Even so, the arm cannot leave its joint range: the limit constraints in the physics still hold it. What changes is that the setpoint may sit outside the range and keep pushing, rather than being redirected along the limit surface. The ablation therefore tests whether that redirection causes the failure. It does not test whether the arm can pass through a limit.

## E3 and E4 Where the runs separate

`results/article/qpos_traces.csv`, `results/article/divergence.json`

- Separation threshold: 0.01 rad on `||q_1e-3 - q_1e-2||`.
- Crossed at step 1401, t = 2.802 s, at a separation of 0.010121 rad.
- Dominant joint at that moment: joint4, 0.008243 rad of the total.
- cond(J) there: 84.5 for 1e-3 against 80.3 for 1e-2. Both are small. The arm is nowhere near a singularity when the paths split.
- Distance from home there: 1.1741 against 1.1640 rad, still almost identical.
- Target radius there: 0.6988 m.
- First clip in the failing run: step 1860, t = 3.720 s, on joint4.
- **Divergence precedes the first clip: True**, by 459 steps.

The third trace in `qpos_traces.csv`, damping 1e-3 at gain 5, is the control: same damping as the failing run, healthy outcome. It shows the split is not a property of the damping value on its own.

## E5 The 2D sweep

`results/article/sweep_2d.csv`, 42 runs, none diverged or raised.

RMS position error, in metres.

| damping | Kn 0 | Kn 0.5 | Kn 1 | Kn 2 | Kn 5 | Kn 10 |
| --- | --- | --- | --- | --- | --- | --- |
| `1e-04` | 0.01451 | 0.01449 | 0.01362 | 0.17590 | 0.01207 | 0.01106 |
| `3e-04` | 0.01364 | 0.01150 | 0.01105 | 0.01065 | 0.00911 | 0.00887 |
| `5e-04` | 0.17580 | 0.01049 | 0.00982 | 0.00891 | 0.00886 | 0.00887 |
| `1e-03` | 0.17572 | 0.00888 | 0.00886 | 0.00887 | 0.00888 | 0.00891 |
| `3e-03` | 0.17538 | 0.00888 | 0.00890 | 0.00894 | 0.00905 | 0.00924 |
| `5e-03` | 0.17483 | 0.00892 | 0.00898 | 0.00908 | 0.00938 | 0.00989 |
| `1e-02` | 0.00888 | 0.00909 | 0.00930 | 0.00970 | 0.01090 | 0.01275 |

### Where the band is, and where it is not

- At gain 0 the band covers `5e-04`, `1e-03`, `3e-03`, `5e-03`, exactly the range the committed sweep reports.
- Any gain from 0.5 upward clears it. At `5e-04` the error falls from 0.17580 to 0.01049 m.

### Two failures, not one

The secondary objective does not repair the low-damping regime. At damping `1e-04` the clip count and the conditioning across gains are:

| gain | RMS position [m] | clipped steps | peak cond(J) | max drift [rad] |
| --- | --- | --- | --- | --- |
| 0 | 0.01451 | 12 | 32,681 | 1.400 |
| 0.5 | 0.01449 | 11 | 17,946 | 1.399 |
| 1 | 0.01362 | 10 | 12,474 | 1.399 |
| 2 | 0.17590 | 1147 | 9,252 | 2.673 |
| 5 | 0.01207 | 8 | 57,303 | 1.399 |
| 10 | 0.01106 | 5 | 172,014 | 1.392 |

Clipping persists at every gain, and the conditioning gets *worse* as the gain rises: peak cond(J) goes from 32,681 at gain 0 to 57,303 at gain 5 and 172,014 at gain 10. The low-damping velocity spike and the mid-band posture collapse are two different failures and want two different fixes.

### An isolated failure that is not in the band

| damping | gain | RMS position [m] | clipped steps | max drift [rad] |
| --- | --- | --- | --- | --- |
| `1e-04` | 2 | 0.17590 | 1147 | 2.673 |
| `5e-04` | 0 | 0.17580 | 1145 | 2.673 |
| `1e-03` | 0 | 0.17572 | 1140 | 2.673 |
| `3e-03` | 0 | 0.17538 | 1140 | 2.673 |
| `5e-03` | 0 | 0.17483 | 1140 | 2.673 |

Damping `1e-04` at gain 2 fails the same way, with the same 2.67 rad drift, while both its neighbours in gain are healthy. The failure is not monotone in the gain either: raising the secondary objective does not simply make things safer, it moves where the bad pocket sits.

## E6 Does any of this survive a different reach?

`results/article/radius_robustness.csv`

Everything above rests on one trajectory, reaching 0.70 m from the shoulder. E6 repeats the grid at 6 radii, 126 runs in total, and records the final joint angles as well as the summary.

### Two failures were being counted as one

Separating them was necessary before the radius question could be answered at all. A run can miss the target simply because the pose is out of reach at the peak: it lags through the middle of the trajectory, then comes home. That is the task being impossible, not the solver choosing badly. The failure this study is about ends somewhere else entirely.

The two are cleanly separable by where the run ends. The collapse reaches 2.671 to 2.673 rad from home; every other run, failing or not, stays at or below 1.599. There is nothing in between.

| end state | runs |
| --- | --- |
| healthy | 93 |
| infeasible | 13 |
| collapse | 20 |

### The collapse is one fixed configuration

All 20 collapsed runs, spanning 3 radii, 7 damping values and 3 nullspace gains, end within 0.0079 rad of each other. This is a stronger statement than the matching drift norm: two different postures can share a norm, and these do not merely share one, they are the same posture.

| joint | mean final angle [rad] | std [rad] | limit |
| --- | --- | --- | --- |
| joint1 | 0.0001 | 0.00069 | ±2.8973 |
| joint2 | 1.7472 | 0.00019 | ±1.7628 |
| joint3 | 0.0066 | 0.00042 | ±2.8973 |
| joint4 | -0.0678 | 0.00004 | [-3.0718, -0.0698] |
| joint5 | 0.0050 | 0.00242 | ±2.8973 |
| joint6 | 0.2188 | 0.00068 | [-0.0175, 3.7525] |
| joint7 | -0.7824 | 0.00289 | ±2.8973 |

joint4 comes to rest at -0.0678 rad against an upper limit of -0.0698, and joint2 at 1.7472 against 1.7628. The arm ends folded over onto two of its limits, and it is the same fold every time. Healthy runs are equally consistent in the other direction: all 93 of them finish within 0.00118 rad of the home posture.

### The band does not survive unchanged, and this matters

Collapsed runs out of 7 damping values, by radius and gain:

| peak radius [m] | Kn 0 | Kn 2 | Kn 5 |
| --- | --- | --- | --- |
| 0.66 | 0 | 0 | 0 |
| 0.68 | 0 | 0 | 0 |
| 0.69 | 0 | 0 | 0 |
| 0.70 | 4 | 1 | 0 |
| 0.71 | 5 | 0 | 0 |
| 0.72 | 6 | 2 | 2 |

- radius 0.70, gain 0: `5e-04`, `1e-03`, `3e-03`, `5e-03`
- radius 0.71, gain 0: `5e-04`, `1e-03`, `3e-03`, `5e-03`, `1e-02`
- radius 0.72, gain 0: `1e-04`, `3e-04`, `5e-04`, `3e-03`, `5e-03`, `1e-02`
- radius 0.70, gain 2: `1e-04`
- radius 0.72, gain 2: `1e-04`, `5e-04`
- radius 0.72, gain 5: `1e-04`, `3e-04`

The collapse does not exist below 0.70 m. It appears at 0.70 and widens with reach: four of seven damping values at 0.70, five at 0.71, six at 0.72, all at gain 0. So the band reported for 0.70 is not a coincidence of that one path, but neither is it a fixed interval of damping. It is the boundary layer of a threshold in task difficulty, and the damping decides which side of the fold the arm comes down on only while the task sits near that threshold.

The secondary objective is not a general cure either. It clears every collapse at 0.70 by gain 5, but at 0.72 gain 5 still collapses at damping `1e-04` and `3e-04`, and at 0.70 gain 2 introduces a collapse at `1e-04` that gain 0 does not have. Raising the gain moves the vulnerable region rather than removing it.

**What can honestly be claimed:** near the edge of the orientation-constrained workspace there is a regime where the damping chooses between two outcomes that are not on a continuum, and the losing outcome is one specific posture the arm cannot leave. What cannot be claimed is that damped least squares has a failure band at some fixed interval of lambda.

## Environment

- mujoco 3.12.0
- mink 1.3.0
- numpy 2.5.2
- Python 3.12.4
- mujoco_menagerie commit `da76818e269b82289eba39808e2fb91d679d6994`
- trajectory: 0.587 m to 0.700 m from the shoulder and back
- 3000 steps per run, timestep 0.002 s, duration 6 s
- integration_dt 1.0 throughout

## Wall clock

| experiment | seconds |
| --- | --- |
| E1_clip_forensics | 1.78 |
| E2_ablation | 1.45 |
| E3_traces | 1.31 |
| E4_divergence | 0.00 |
| E5_sweep_2d | 13.05 |
| E6_radius_robustness | 38.69 |
| **total** | **56.28** |

## Replication of the externally supplied result

`results/nullspace_removes_failure_band.csv` was produced outside this session. Treated as a replication, its gain-0 columns agree with the committed `results/damping_summary.csv` over all 11 damping values: maximum relative difference 0.000e+00 in RMS position error, 0.000e+00 in peak cond(J), and 0 in clipped steps. The environments are equivalent, so its gain-5 column can be taken as data.

E5 reproduces that gain-5 column independently. Over the 7 damping values the two runs share, the maximum relative difference in RMS position error is 3.634e-06 and the clip counts differ by at most 0.

