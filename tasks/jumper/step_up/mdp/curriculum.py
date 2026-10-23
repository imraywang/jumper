"""Step height curriculum: up a row when the episode ended on the step, down when it never got past the face."""
from __future__ import annotations

import torch

from .rewards import on_step, progress


def step_levels(env, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    terrain = env.scene.terrain
    up = on_step(env)[env_ids] > 0.5
    down = (progress(env)[env_ids] < 0.40 - 0.10) & ~up
    if env.common_step_counter == 0:
        up = torch.zeros_like(up)
        down = torch.zeros_like(down)
    terrain.update_env_origins(env_ids, up, down)
    levels = terrain.terrain_levels.float()
    return {"mean": levels.mean(), "max": levels.max()}
