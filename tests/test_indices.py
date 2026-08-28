"""Index resolution. Everything downstream trusts these ids, so they are checked
against the model rather than against hardcoded integers."""

import mujoco
import numpy as np
import pytest

from diffik import model as model_mod
from diffik.model import RobotHandles


def test_arm_index_arrays_have_seven_entries(handles: RobotHandles) -> None:
    assert handles.dof_ids.shape == (7,)
    assert handles.qpos_ids.shape == (7,)
    assert handles.act_ids.shape == (7,)
    assert handles.q_home.shape == (7,)


def test_indices_are_within_model_bounds(handles: RobotHandles) -> None:
    model = handles.model
    assert np.all((handles.dof_ids >= 0) & (handles.dof_ids < model.nv))
    assert np.all((handles.qpos_ids >= 0) & (handles.qpos_ids < model.nq))
    assert np.all((handles.act_ids >= 0) & (handles.act_ids < model.nu))
    assert 0 <= handles.site_id < model.nsite
    assert 0 <= handles.key_id < model.nkey
    assert 0 <= handles.mocap_id < model.nmocap


def test_indices_are_unique(handles: RobotHandles) -> None:
    for ids in (handles.dof_ids, handles.qpos_ids, handles.act_ids):
        assert len(np.unique(ids)) == len(ids)


def test_dof_and_qpos_ids_come_from_the_named_arm_joints(
    handles: RobotHandles,
) -> None:
    """dofadr and qposadr are read from the model, never assumed equal. They do
    coincide for the Panda because all seven arm joints are hinges."""
    model = handles.model
    for i, name in enumerate(model_mod.ARM_JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint_id >= 0, f"joint {name!r} missing"
        assert model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
        assert handles.dof_ids[i] == model.jnt_dofadr[joint_id]
        assert handles.qpos_ids[i] == model.jnt_qposadr[joint_id]


def test_actuators_drive_the_arm_joints_in_order(handles: RobotHandles) -> None:
    """data.ctrl[act_ids] = q[qpos_ids] is only correct if the two orderings
    line up joint for joint."""
    model = handles.model
    for act_id, joint_name in zip(handles.act_ids, model_mod.ARM_JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        assert model.actuator_trnid[act_id, 0] == joint_id


def test_fingers_are_excluded_from_the_arm_indices(handles: RobotHandles) -> None:
    """The gripper shares the model with the arm. mj_jacSite fills all nv
    columns, so the finger dofs must not leak into dof_ids."""
    model = handles.model
    for name in ("finger_joint1", "finger_joint2"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint_id >= 0
        assert model.jnt_dofadr[joint_id] not in handles.dof_ids


def test_site_belongs_to_the_hand_body(handles: RobotHandles) -> None:
    model = handles.model
    hand_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, model_mod.HAND_BODY_NAME
    )
    assert model.site_bodyid[handles.site_id] == hand_id
    assert np.allclose(model.site_pos[handles.site_id], model_mod.SITE_POS)


def test_target_is_a_jointless_mocap_body_under_worldbody(
    handles: RobotHandles,
) -> None:
    model = handles.model
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, model_mod.MOCAP_BODY_NAME
    )
    assert model.body_mocapid[body_id] == handles.mocap_id
    assert model.body_jntnum[body_id] == 0
    assert model.body_parentid[body_id] == 0


def test_q_home_matches_the_keyframe(handles: RobotHandles) -> None:
    model = handles.model
    expected = model.key_qpos[handles.key_id][handles.qpos_ids]
    assert np.allclose(handles.q_home, expected)


def test_reset_to_home_puts_qpos_and_site_in_sync(handles: RobotHandles) -> None:
    """mj_forward has to run after the keyframe reset, otherwise site_xpos is
    stale. A displaced site that snaps back after mj_forward proves the reset
    already did it."""
    model_mod.reset_to_home(handles)
    assert np.allclose(handles.data.qpos[handles.qpos_ids], handles.q_home)

    before = handles.data.site_xpos[handles.site_id].copy()
    handles.data.qpos[handles.qpos_ids] += 0.2
    mujoco.mj_forward(handles.model, handles.data)
    assert not np.allclose(handles.data.site_xpos[handles.site_id], before)

    model_mod.reset_to_home(handles)
    assert np.allclose(handles.data.site_xpos[handles.site_id], before)


def test_sync_target_to_site_zeroes_the_pose_error(handles: RobotHandles) -> None:
    data = handles.data
    model_mod.sync_target_to_site(handles)

    assert np.allclose(data.mocap_pos[handles.mocap_id], data.site_xpos[handles.site_id])

    site_quat = np.zeros(4)
    mujoco.mju_mat2Quat(site_quat, data.site_xmat[handles.site_id])
    # Quaternions are equal up to sign, so compare the relative rotation angle.
    relative = np.zeros(4)
    site_quat_inv = np.zeros(4)
    mujoco.mju_negQuat(site_quat_inv, site_quat)
    mujoco.mju_mulQuat(relative, data.mocap_quat[handles.mocap_id], site_quat_inv)
    rotation = np.zeros(3)
    mujoco.mju_quat2Vel(rotation, relative, 1.0)
    assert np.linalg.norm(rotation) < 1e-9


def test_missing_scene_file_is_reported(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        model_mod.load(tmp_path / "does_not_exist.xml")


def test_missing_entity_names_itself(handles: RobotHandles) -> None:
    with pytest.raises(ValueError, match="no_such_site"):
        model_mod._require_id(
            handles.model, mujoco.mjtObj.mjOBJ_SITE, "no_such_site"
        )
