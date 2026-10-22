"""A takeout bag on the floor, its handle flopped over the side toward the robot.

The fetch was tuned on the mug's handle (tools/fetch/mug.py): an outer bar the jaws
pinch across the mouth, and a lower arm that hangs on the finger once the body
pitches up. A soft bag whose strap has fallen against its side presents the same
loop, so the bag reuses the mug's handle exactly -- the same bars, the same
distance off the wall, the same height above the floor -- and only the body
changes: a box of kraft paper rather than a cylinder. The face toward the robot
sits where the mug's wall was, so every number the pilot aims with still holds.
"""
from __future__ import annotations

import os
from typing import Any

import mujoco

from mug import (
    BAR,
    BODY_HALF_H,
    BODY_R,
    GRIP_FRICTION,
    GRIP_PRIORITY,
    GRIP_SOLREF,
    MUG_X,
    MUG_Y,
    handle_boxes,
)
from tasks.jumper.five_foot import objects as O

#: The body: 80 mm deep (the face toward the robot where the mug's wall was),
#: 130 mm wide and 100 mm tall -- a bag for one meal, folded over at the top.
HALF = (BODY_R, 0.065, BODY_HALF_H)
MASS = float(os.environ.get("FETCH_BAG_MASS", "0.075"))   # kg: the mug's mass, the hold it was tuned for
BAG_YAW = 0.0

KRAFT = [0.76, 0.58, 0.38, 1.0]
FOLD = [0.66, 0.49, 0.31, 1.0]
STRAP = [0.93, 0.93, 0.90, 1.0]
LABEL = [0.85, 0.16, 0.12, 1.0]

#: What the claw aims at: the mug's own feature, since the handle is the mug's.
FEATURE = {"offset": (-(BODY_R + 0.046), 0.0), "half_x": BAR, "half_y": BAR, "hold": 0.026, "gap": 0.025}


def _look(body: Any) -> None:
    """The fold across the top and a label on the face, drawn and colliding with nothing."""
    hx, hy, hz = HALF
    body.add_geom(name="bag_fold", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx * 0.6, hy * 1.01, 0.008],
                  pos=[0.0, 0.0, hz + 0.004], rgba=FOLD, contype=0, conaffinity=0, group=2, mass=0.0)
    body.add_geom(name="bag_label", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.001, hy * 0.55, hz * 0.35],
                  pos=[-hx - 0.001, 0.0, hz * 0.25], rgba=LABEL, contype=0, conaffinity=0, group=2, mass=0.0)


def bag_spec() -> Any:
    """The bag's spec: the body box, the mug's strap and the drawn-only details."""
    spec = O._prop_spec()
    body = O._free_body(spec, "bag")
    O._prop_geom(body, name="bag_body", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(HALF),
                 pos=[0.0, 0.0, 0.0], rgba=KRAFT, mass=MASS - 0.024)
    boxes = handle_boxes()
    for i, (pos, half) in enumerate(boxes):
        O._prop_geom(body, name=f"bag_strap_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=half, pos=pos, rgba=STRAP, mass=0.024 / len(boxes),
                     friction=list(GRIP_FRICTION), priority=GRIP_PRIORITY, solref=list(GRIP_SOLREF))
    _look(body)
    return spec


def apply_bag(env_cfg: Any) -> None:
    """The bag in front of the robot, where the mug would stand."""
    env_cfg.sim.nconmax = max(int(env_cfg.sim.nconmax or 0), O.OBJECTS_NCONMAX)
    env_cfg.sim.njmax = max(int(env_cfg.sim.njmax or 0), O.OBJECTS_NJMAX)
    env_cfg.sim.mujoco.cone = O.PROP_CONE
    env_cfg.sim.mujoco.impratio = O.PROP_IMPRATIO
    from math import cos, sin

    from mjlab.entity import EntityCfg

    props = {"bag": EntityCfg(spec_fn=bag_spec, init_state=EntityCfg.InitialStateCfg(
        pos=(MUG_X, MUG_Y, BODY_HALF_H), rot=(cos(BAG_YAW / 2), 0.0, 0.0, sin(BAG_YAW / 2))))}
    env_cfg.scene.entities.update(props)
    O._place_props(env_cfg, props)
    O._place_the_robot(env_cfg)
