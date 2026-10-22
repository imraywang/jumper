"""A pair of slippers held together by a hanger hook, as shops sell them.

The hook is the mug's handle (tools/fetch/mug.py) to the millimetre -- the same bars,
the same distance off the "wall", the same height -- so `fetch.py` grasps it as it
grasps the mug, and the pair hangs from the claw the way the mug does. The rest is
the pair itself: two soles toes-first toward the robot, either side of the hook,
each with a strap across its front, standing on the floor.

The frame is the mug's: the origin 50 mm above the floor, the hook toward -x.
"""
from __future__ import annotations

from typing import Any

import mujoco
from mug import BODY_HALF_H, GRIP_FRICTION, GRIP_PRIORITY, GRIP_SOLREF, handle_boxes

from tasks.jumper.five_foot import objects as O

MASS = 0.08                       # kg: a pair of light slides, about the mug's 75 g
SOLE_HALF = (0.11, 0.045, 0.007)  # 22 x 9 cm, 14 mm thick
SOLE_X = 0.11                     # the soles' centre along x: the toes 2.6 cm behind the hook's inner end.
                                  # At 0.06 the toes stood 1.4 cm ahead of the hook, the jaw met them on
                                  # every creep, and missed approaches pushed the pair 1.5 m (story_take)
SOLE_Y = 0.062                    # either side of the hook
STRAP_X = 0.0                     # the strap across each sole's front half
STRAP_H = 0.035
STRAP_T = 0.006
#: The pair stands on a spacer block between the soles, so the hook's frame sits where
#: the mug's does: 50 mm up, with the soles' undersides on the floor.
SPACER_HALF = (0.03, 0.015, 0.03)

SOLE_RGBA = [0.35, 0.55, 0.78, 1.0]
STRAP_RGBA = [0.92, 0.92, 0.95, 1.0]
HOOK_RGBA = [0.95, 0.80, 0.20, 1.0]

FEATURE = {"offset": (-(0.040 + 0.046), 0.0), "half_x": 0.004, "half_y": 0.004, "hold": 0.026, "gap": 0.025}


def slippers_spec() -> Any:
    """The pair: soles and straps that collide, the mug's hook between them."""
    spec = O._prop_spec()
    body = O._free_body(spec, "slippers")
    z_sole = -BODY_HALF_H + SOLE_HALF[2]
    hook_mass = 0.012
    rest = MASS - hook_mass
    for side in (-1.0, 1.0):
        y = side * SOLE_Y
        O._prop_geom(body, name=f"sole_{'l' if side > 0 else 'r'}", type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=list(SOLE_HALF), pos=[SOLE_X, y, z_sole], rgba=SOLE_RGBA, mass=rest * 0.4)
        top = z_sole + SOLE_HALF[2] + STRAP_H
        for k, dy in enumerate((-SOLE_HALF[1] + STRAP_T, SOLE_HALF[1] - STRAP_T)):
            O._prop_geom(body, name=f"strap_post_{'l' if side > 0 else 'r'}{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                         size=[0.02, STRAP_T, STRAP_H / 2], pos=[SOLE_X + STRAP_X, y + dy, top - STRAP_H / 2],
                         rgba=STRAP_RGBA, mass=rest * 0.025)
        O._prop_geom(body, name=f"strap_top_{'l' if side > 0 else 'r'}", type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=[0.02, SOLE_HALF[1], STRAP_T], pos=[SOLE_X + STRAP_X, y, top], rgba=STRAP_RGBA,
                     mass=rest * 0.05)
    O._prop_geom(body, name="spacer", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(SPACER_HALF),
                 pos=[0.0, 0.0, -BODY_HALF_H + SPACER_HALF[2]], rgba=STRAP_RGBA, mass=0.0001)
    for i, (pos, half) in enumerate(handle_boxes()):
        O._prop_geom(body, name=f"hook_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=pos,
                     rgba=HOOK_RGBA, mass=hook_mass / 3,
                     friction=list(GRIP_FRICTION), priority=GRIP_PRIORITY, solref=list(GRIP_SOLREF))
    return spec
