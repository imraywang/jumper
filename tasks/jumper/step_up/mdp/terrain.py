"""One step per patch: flat ground, then a box from FACE ahead of the spawn to the far edge.

The curriculum generator gives row r the difficulty r / (rows - 1), so a patch's
step height is a function of its row alone, and its face distance of its column
(one sub-terrain type per column, each with its own `face`). `step_geometry`
recovers both for every env from its terrain level and type, which is what the
rewards and the curriculum read: no ray cast, no lookup table built at generation.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane

#: Where the robot spawns in its patch, metres from the patch's low corner.
SPAWN = (0.5, 0.6)


@dataclass(kw_only=True)
class StepTerrainCfg(SubTerrainCfg):
    step_height_range: tuple[float, float]
    """Step height at difficulty 0 and 1, metres."""
    face: float
    """The step's near face, metres ahead of the spawn along +x."""

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        h = self.step_height_range[0] + difficulty * (self.step_height_range[1] - self.step_height_range[0])
        geoms = [TerrainGeometry(geom=make_plane(body, self.size, 0.0, center_zero=False)[0],
                                 color=(0.45, 0.47, 0.50, 1.0))]
        if h > 1e-4:
            x0 = SPAWN[0] + self.face
            box = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=((self.size[0] - x0) / 2, self.size[1] / 2, h / 2),
                                pos=((self.size[0] + x0) / 2, self.size[1] / 2, h / 2))
            geoms.append(TerrainGeometry(geom=box, color=(0.62, 0.46, 0.30, 1.0)))
        return TerrainOutput(origin=np.array([SPAWN[0], SPAWN[1], 0.0]), geometries=geoms)


def step_geometry(env) -> tuple[torch.Tensor, torch.Tensor]:
    """(height, face) of every env's step, from its terrain level and type."""
    terrain = env.scene.terrain
    gen = terrain.cfg.terrain_generator
    cfgs = list(gen.sub_terrains.values())
    lo, hi = gen.difficulty_range
    rows = max(gen.num_rows - 1, 1)
    diff = lo + (hi - lo) * terrain.terrain_levels.float() / rows
    h_lo = torch.tensor([c.step_height_range[0] for c in cfgs], device=env.device)
    h_hi = torch.tensor([c.step_height_range[1] for c in cfgs], device=env.device)
    face = torch.tensor([c.face for c in cfgs], device=env.device)
    t = terrain.terrain_types
    return h_lo[t] + diff * (h_hi[t] - h_lo[t]), face[t]


def step_obs(env, mirror_kind: str = "scalar") -> torch.Tensor:
    """The critic's view of the step: its height, and how far ahead of the base its face is.

    Both unchanged by the left-right mirror (`mirror_kind`, read by common/mdp/symmetry.py)."""
    del mirror_kind
    h, face = step_geometry(env)
    x = env.scene["robot"].data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return torch.stack([h, face - x], dim=1)
