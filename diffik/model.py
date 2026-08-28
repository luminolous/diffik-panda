"""Model loading and one-time index resolution.

Names recorded from the vendored scene/franka_emika_panda/panda.xml so that
nothing downstream has to guess them:

    joints      joint1 .. joint7          hinge, the arm
                finger_joint1             slide, gripper
                finger_joint2             slide, gripper
    actuators   actuator1 .. actuator7    position servo, one per arm joint
                actuator8                 gripper, driven through tendon "split"
    bodies      link0 .. link7, hand, left_finger, right_finger
    keyframe    home                      qpos has 9 entries, ctrl has 8
    sites       none

The vendored model defines no site at all, so there is nothing to point
mj_jacSite at. mjx_panda.xml places its "gripper" site at pos="0 0 0.1" in the
hand frame; this module adds an end-effector site at that same offset through
MjSpec. It is done here rather than in scene/panda_ik.xml because MJCF cannot
add a child to a body that came from another file, and the vendored model must
not be edited.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

# Arm joints, in Jacobian column order. The fingers are deliberately excluded:
# they move the gripper, not the end-effector pose the IK is solving for.
ARM_JOINT_NAMES: tuple[str, ...] = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
)

# Position actuators for the arm. actuator8 drives the gripper and is not part
# of the IK command.
ARM_ACTUATOR_NAMES: tuple[str, ...] = (
    "actuator1",
    "actuator2",
    "actuator3",
    "actuator4",
    "actuator5",
    "actuator6",
    "actuator7",
)

HAND_BODY_NAME = "hand"
MOCAP_BODY_NAME = "target"
KEYFRAME_NAME = "home"

SITE_NAME = "ee_site"
# Grasp centre between the fingers, expressed in the hand frame. Matches the
# "gripper" site of mjx_panda.xml.
SITE_POS: tuple[float, float, float] = (0.0, 0.0, 0.1)


@dataclass
class RobotHandles:
    """Everything resolved once at startup, so the control loop never looks up a
    name. Passing this around keeps mj_name2id out of the hot path."""

    model: mujoco.MjModel
    data: mujoco.MjData
    dof_ids: NDArray[np.int32]
    """Velocity-space indices: Jacobian columns, qvel entries, length 7."""
    qpos_ids: NDArray[np.int32]
    """Configuration-space indices, length 7. Equal to dof_ids for the Panda
    because every arm joint is a hinge, but the two spaces diverge for ball and
    free joints, so they are kept apart on purpose."""
    act_ids: NDArray[np.int32]
    site_id: int
    mocap_id: int
    key_id: int
    q_home: NDArray[np.float64]
    """Arm entries of the home keyframe, the reference posture for the
    nullspace term."""


def _require_id(model: mujoco.MjModel, obj: mujoco.mjtObj, name: str) -> int:
    """Resolve a name to an id, or fail with a message that says what is
    missing. A silent -1 turns into a confusing out-of-bounds error much later."""
    obj_id = mujoco.mj_name2id(model, obj, name)
    if obj_id < 0:
        kind = obj.name.removeprefix("mjOBJ_").lower()
        raise ValueError(f"{kind} {name!r} not found in the model")
    return obj_id


def _compile_with_site(scene_path: Path) -> mujoco.MjModel:
    """Compile the scene and attach the end-effector site to the hand body."""
    spec = mujoco.MjSpec.from_file(str(scene_path))

    body = spec.body(HAND_BODY_NAME)
    if body is None:
        raise ValueError(
            f"body {HAND_BODY_NAME!r} not found in {scene_path}; the vendored "
            "model layout changed"
        )

    if not any(site.name == SITE_NAME for site in spec.sites):
        site = body.add_site()
        site.name = SITE_NAME
        site.pos = np.asarray(SITE_POS, dtype=np.float64)
        # Small and translucent: it marks the controlled frame without hiding
        # the gripper behind it.
        site.size = np.array([0.008, 0.008, 0.008])
        site.rgba = np.array([0.2, 0.9, 0.2, 0.6])

    return spec.compile()


def load(scene_path: str | Path) -> RobotHandles:
    """Compile the scene, resolve every index by name, and reset to home."""
    scene_path = Path(scene_path)
    if not scene_path.is_file():
        raise FileNotFoundError(f"scene file not found: {scene_path}")

    model = _compile_with_site(scene_path)
    data = mujoco.MjData(model)

    joint_ids = [
        _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in ARM_JOINT_NAMES
    ]
    dof_ids = np.asarray([model.jnt_dofadr[j] for j in joint_ids], dtype=np.int32)
    qpos_ids = np.asarray([model.jnt_qposadr[j] for j in joint_ids], dtype=np.int32)
    act_ids = np.asarray(
        [
            _require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ARM_ACTUATOR_NAMES
        ],
        dtype=np.int32,
    )

    site_id = _require_id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
    key_id = _require_id(model, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME_NAME)

    mocap_body_id = _require_id(model, mujoco.mjtObj.mjOBJ_BODY, MOCAP_BODY_NAME)
    mocap_id = int(model.body_mocapid[mocap_body_id])
    if mocap_id < 0:
        raise ValueError(
            f"body {MOCAP_BODY_NAME!r} exists but is not a mocap body; it needs "
            'mocap="true" and must be a direct child of worldbody'
        )

    q_home = np.array(model.key_qpos[key_id][qpos_ids], dtype=np.float64)

    handles = RobotHandles(
        model=model,
        data=data,
        dof_ids=dof_ids,
        qpos_ids=qpos_ids,
        act_ids=act_ids,
        site_id=site_id,
        mocap_id=mocap_id,
        key_id=key_id,
        q_home=q_home,
    )
    reset_to_home(handles)
    return handles


def reset_to_home(handles: RobotHandles) -> None:
    """Reset to the home keyframe. mj_forward is required: without it site_xpos
    still holds the previous configuration and the first error is garbage."""
    mujoco.mj_resetDataKeyframe(handles.model, handles.data, handles.key_id)
    mujoco.mj_forward(handles.model, handles.data)


def sync_target_to_site(handles: RobotHandles) -> None:
    """Place the mocap target exactly on the current site pose, so the first
    control step sees near-zero error and the arm does not jump."""
    data = handles.data
    data.mocap_pos[handles.mocap_id] = data.site_xpos[handles.site_id]
    mujoco.mju_mat2Quat(
        data.mocap_quat[handles.mocap_id],
        data.site_xmat[handles.site_id],
    )
