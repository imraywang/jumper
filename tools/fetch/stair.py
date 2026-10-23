"""A probe step for the hop: one box of height STEP_H whose near face is STEP_X ahead of the start."""
from __future__ import annotations

import os
from typing import Any

STEP_H = float(os.environ.get("STEP_H", "0.08"))
STEP_X = float(os.environ.get("STEP_X", "0.35"))
STEP_DEPTH = 0.6


def decorate(spec: Any) -> None:
    import mujoco

    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[STEP_DEPTH / 2, 0.6, STEP_H / 2],
                            pos=[STEP_X + STEP_DEPTH / 2, 0.0, STEP_H / 2], rgba=[0.6, 0.45, 0.3, 1], group=2, mass=0.0)


def apply_stair(env_cfg: Any) -> None:
    from scenes.registry import _install_spec_edits

    env_cfg.sim.nconmax = max(int(env_cfg.sim.nconmax or 0), 1024)
    env_cfg.sim.njmax = max(int(env_cfg.sim.njmax or 0), 8192)
    _install_spec_edits(env_cfg.scene, None, decorate)
