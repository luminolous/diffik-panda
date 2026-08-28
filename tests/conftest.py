"""Shared fixtures. Compiling the Panda takes a moment, so the handles are built
once per session and reset between tests."""

from pathlib import Path

import pytest

from diffik import model as model_mod

SCENE_PATH = Path(__file__).resolve().parents[1] / "scene" / "panda_ik.xml"


@pytest.fixture(scope="session")
def handles() -> model_mod.RobotHandles:
    return model_mod.load(SCENE_PATH)


@pytest.fixture(autouse=True)
def _home(handles: model_mod.RobotHandles) -> None:
    """Every test starts from the home keyframe, whatever the previous one did."""
    model_mod.reset_to_home(handles)
