"""What the step changes about the dash's rewards: where the ground is, and the goal."""
from __future__ import annotations

import torch

from ...jump.mdp.reference import ref_state
from ...jump.mdp.rewards import _foot_body_ids
from .terrain import step_geometry

#: The ground under the base ramps from the floor to the step's top over this
#: span of base x around the face (m): a hop crossing the edge sees its target
#: height rise as it goes over, not jump by h at one instant.
RAMP = (-0.05, 0.20)


def _local(env, robot):
    return robot.data.root_link_pos_w - env.scene.env_origins


def ground_under_base(env, robot) -> torch.Tensor:
    h, face = step_geometry(env)
    x = _local(env, robot)[:, 0]
    s = ((x - face - RAMP[0]) / (RAMP[1] - RAMP[0])).clamp(0.0, 1.0)
    return h * s


def ref_base_height_stepped(env, command_name: str = "jump", sigma_sq: float = 0.0025) -> torch.Tensor:
    """jump's `ref_base_height`, measured from the ground under the base rather than the floor."""
    _, ref, _, _ = ref_state(env, command_name)
    robot = env.scene["robot"]
    z = _local(env, robot)[:, 2] - ground_under_base(env, robot)
    return torch.exp(-((z - ref["base_pos"][:, 2]) ** 2) / sigma_sq)


def on_step(env) -> torch.Tensor:
    """1 while every foot is past the face and on the top, and the base stands over it."""
    h, face = step_geometry(env)
    robot = env.scene["robot"]
    feet = robot.data.body_link_pos_w[:, _foot_body_ids(env)] - env.scene.env_origins.unsqueeze(1)
    past = (feet[:, :, 0] > face.unsqueeze(1) + 0.01).all(dim=1)
    up = (feet[:, :, 2] > h.unsqueeze(1) - 0.02).all(dim=1)
    standing = _local(env, robot)[:, 2] > h + 0.06
    return (past & up & standing).float()


def progress(env) -> torch.Tensor:
    """How far past the face the rearmost foot is, capped at the goal: a gradient toward it."""
    _, face = step_geometry(env)
    robot = env.scene["robot"]
    feet = robot.data.body_link_pos_w[:, _foot_body_ids(env)] - env.scene.env_origins.unsqueeze(1)
    rear = feet[:, :, 0].min(dim=1).values
    return (rear - face).clamp(-0.40, 0.02) + 0.40
