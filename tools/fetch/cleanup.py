"""Tidy up: fetch items one after the other and set each down where it belongs.

`job="tidy"` (the default) is the bag and the mug into the bin; `job="slippers"` is the
pair of slippers to the bedside, set down facing the bed.

    FETCH_LIFT=1 .venv/bin/python tools/fetch/run.py --pilot tools/fetch/cleanup.py \\
        --task jumper.five_foot --cleanup --scene beach --headless --steps 24000 --video ...

Each pick is `fetch.py`'s own, unchanged, from the approach to the lift. When an item
is up (fetch's `carry`) this pilot takes over: it raises the nose further, walks
straight back until the item is level with the bin, strafes across until it is over
it, sets it down on the bin's floor, opens the claw and twists the jaw out of the
handle sideways, backs off and stows the arm, and hands the next item to a fresh
fetch started at `approach`. Straight back before across, rather than
`fetch.walk_to`'s across-then-along: the bin stands beside the pickup line, and
crossing first would walk the body into it.

Every target is the item's own measured position, not the base's: the mug hangs
9 cm nearer the body than the bag (cleanup_v1: 0.28 against 0.36 m ahead of the
base), so one drop spot for both put the mug against the bin's near wall.

The handle hangs on the finger as well as in the pinch (mug.py: the lower arm meets
the pads from below), so opening the claw does not let go of it: in cleanup_v1 the
opened mug stayed on the jaw and was dragged out of the bin on the retreat. The
loop is open only at its sides, so the jaw leaves it sideways, by the body twist,
once the item stands on the floor.

Ground truth stands in for perception, as in `fetch.py`.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from fetch import LIFT, LIFT_TO, TURN, TWIST_SIGN, _wrap, _yaw
from fetch import make_pilot as fetch_pilot
from mug import BODY_HALF_H as M_HALF_H

#: The litter, in the order it is picked up: (scene name, floor position of the body
#: centre). Both stand at the mug's own distance, so `fetch.py`'s approach is the one
#: it was tuned on; the mug goes 0.81 m to the right of the bag. At 0.36 m the bag's
#: missed approaches pushed it beside the mug and the claw came up with the mug
#: (cleanup_v2).
ITEMS: tuple[tuple[str, tuple[float, float]], ...] = (("bag", (0.587, 0.064)), ("mug", (0.587, -0.75)))
#: The slippers, where the alarm film (tools/fetch/story.py) leaves them: off the rug's
#: corner, clear of both pickup lines and of the map's plate and basket.
SLIPPERS_XY = (-0.25, -0.95)

#: An item counts as lifted when its centre is this far above where it stood.
LIFTED_DZ = 0.03
RETRIES = 2                      # fresh approaches for an item the claw did not come up with
MAX_BACK = 1.5                   # m: the furthest the base walks along its axis looking for the goal's line

#: The bin: beside the pickup line, its near wall far enough ahead of a base that
#: holds an item over it that the front legs stay clear.
BIN_XY = (0.155, 0.606)
BIN_HALF = (0.09, 0.17)          # 18 x 34 cm inside
#: Where in the bin each item goes, across it from the centre: the first one in at the
#: far end, so the second neither lands on it nor is carried over it. Both aimed at the
#: centre, the mug came down on the bag, and opening the claw threw the bag out (home_v1).
SLOT_Y = {"bag": +0.08, "mug": -0.09}

#: Floor beside home.map's bed (its side at x = -0.94), where the slippers are set down
#: toes out, for feet coming off the bed's middle.
SLIPPERS_GOAL = (-0.80, 0.30)

#: What a pilot does: the items in the order they are picked, where each one goes, the
#: heading the drop is made at (pi: facing the bed, the robot's back to the room), and
#: whether it dances when it is done.
JOBS: dict[str, dict] = {
    "tidy": {"items": ("bag", "mug"), "yaw": 0.0, "celebrate": True,
             "goals": {n: (BIN_XY[0], BIN_XY[1] + SLOT_Y[n]) for n in ("bag", "mug")}},
    "slippers": {"items": ("slippers",), "yaw": math.pi, "celebrate": False,
                 "goals": {"slippers": SLIPPERS_GOAL}},
}
BIN_WALL_H = 0.045               # under a lifted item at CARRY_NOSE, which the strafe carries over the side wall
                                 # (0.06 at LIFT_TO stopped the mug against it, cleanup_v1)
CARRY_NOSE = 0.8                 # the nose for the carry to the bin, raised from fetch's LIFT_TO over RAISE_S
RAISE_S = 1.5
SET_NOSE = -0.9                  # the nose that puts the item down on the bin's floor; -0.5 left the
                                 # mug hanging 5 cm up on the finger when the claw opened (cleanup_v3)
SET_S = 2.0
SET_MAX_S = 3.5                  # stop lowering at this, landed or not
SLIP_TWIST = -0.45               # fetch's twist limit: ~11 cm of jaw sideways, out of the loop. +0.45
                                 # dragged the mug 11 cm across and over the bin's side wall (cleanup_v3)
SLIP_S = 1.5
ON_TARGET = 0.03                 # m: the item's centre this close to the bin's, along each axis
BIN_WALL_T = 0.006
BIN_RGBA = [0.20, 0.42, 0.30, 1.0]
BIN_FLOOR_RGBA = [0.15, 0.30, 0.22, 1.0]

WALK = 0.5                       # stick travel: above the policy's ~0.4 dead band
RETREAT_S = 1.5
STOW_S = 1.0


def _decorate(spec: Any) -> None:
    import mujoco

    bx, by = BIN_XY
    hx, hy = BIN_HALF
    t, h = BIN_WALL_T, BIN_WALL_H
    # The floor is drawn only, as room.py's tray mat is: a collidable plate's edge
    # catches a foot. The items land on the ground through it.
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, 0.002], pos=[bx, by, 0.002],
                            rgba=BIN_FLOOR_RGBA, group=2, mass=0.0, contype=0, conaffinity=0)
    for dx, dy, sx, sy in ((hx + t, 0.0, t, hy + 2 * t), (-hx - t, 0.0, t, hy + 2 * t),
                           (0.0, hy + t, hx, t), (0.0, -hy - t, hx, t)):
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, h / 2], pos=[bx + dx, by + dy, h / 2],
                                rgba=BIN_RGBA, group=2, mass=0.0)


def apply_cleanup(env_cfg: Any) -> None:
    """The bag and the mug in front of the robot, and the bin beside them."""
    import mug as M
    from bag import bag_spec
    from mjlab.entity import EntityCfg

    from scenes.registry import _install_spec_edits
    from tasks.jumper.five_foot import objects as O

    env_cfg.sim.nconmax = max(int(env_cfg.sim.nconmax or 0), 1024)
    env_cfg.sim.njmax = max(int(env_cfg.sim.njmax or 0), 8192)
    env_cfg.sim.mujoco.cone = O.PROP_CONE
    env_cfg.sim.mujoco.impratio = O.PROP_IMPRATIO
    specs = {"bag": bag_spec, "mug": M.mug_spec}
    props = {name: EntityCfg(spec_fn=specs[name], init_state=EntityCfg.InitialStateCfg(
        pos=(x, y, M.BODY_HALF_H), rot=(1.0, 0.0, 0.0, 0.0))) for name, (x, y) in ITEMS}
    env_cfg.scene.entities.update(props)
    O._place_props(env_cfg, props)
    O._place_the_robot(env_cfg)
    _install_spec_edits(env_cfg.scene, None, _decorate)


def in_bin(p: np.ndarray) -> bool:
    return (abs(p[0] - BIN_XY[0]) < BIN_HALF[0] and abs(p[1] - BIN_XY[1]) < BIN_HALF[1]
            and p[2] < BIN_WALL_H + 0.02)


def placed(p: np.ndarray, goal: tuple[float, float]) -> bool:
    return math.hypot(p[0] - goal[0], p[1] - goal[1]) < 0.08 and p[2] < M_HALF_H + 0.02


def make_pilot(env, player, log=print, job: str = "tidy"):
    from controller import GamepadState

    if not LIFT:
        raise SystemExit("cleanup needs the lifted carry: run with FETCH_LIFT=1")
    robot = env.scene["robot"]
    plan = JOBS[job]
    names = list(plan["items"])
    want_yaw = plan["yaw"]
    st = {"phase": "fetch", "since": 0.0, "item": 0, "turn": 0, "stall_p": None, "stall_t": 0.0, "bump": 0.0,
          "lifted": 0.0, "tries": 0, "retries": 0, "back_from": 0.0}
    inner = [fetch_pilot(env, player, prop=names[0], log=log)]
    last_log = [-1.0]

    def goto(phase, t):
        log(f"\n[cleanup] {t:6.2f}s  {st['phase']} -> {phase}  ({names[st['item']]})")
        st.update(phase=phase, since=t, turn=0, stall_p=None, bump=0.0)

    def item_pos(name):
        return env.scene[name].data.root_link_pos_w[0].cpu().numpy()

    def stick(base_p, t):
        """WALK, raised while the base is not moving, as fetch.py's stick_mag does."""
        if st["stall_p"] is None or np.linalg.norm(base_p[:2] - st["stall_p"]) > 0.015:
            st["stall_p"], st["stall_t"] = base_p[:2].copy(), t
        elif t - st["stall_t"] > 1.0:
            st["bump"] = min(0.3, st["bump"] + 0.05)
            st["stall_t"] = t
        return WALK + st["bump"]

    def heading(yaw, s):
        """fetch.py's turn pulses, toward the job's drop heading."""
        e = _wrap(want_yaw - yaw)
        if st["turn"] == 0 and abs(e) > 0.10:
            st["turn"] = 1 if e > 0 else -1
        elif st["turn"] != 0 and (abs(e) < 0.03 or (e > 0) != (st["turn"] > 0)):
            st["turn"] = 0
        if st["turn"] != 0:
            s["rx"] = TURN * st["turn"] * TWIST_SIGN

    def pilot(t):
        if st["phase"] == "fetch":
            pad = inner[0](t)
            if inner[0].state["name"] != "carry":
                return pad
            item = item_pos(names[st["item"]])
            if item[2] < M_HALF_H + LIFTED_DZ:
                # Up came something else, or nothing: put it down and start this item again.
                log(f"\n[cleanup] {names[st['item']]} was not lifted (z {item[2]:.3f}); "
                    f"{'approaching again' if st['retries'] < RETRIES else 'skipping it'}")
                st["retries"] += 1
                if st["retries"] <= RETRIES:
                    inner[0] = fetch_pilot(env, player, prop=names[st["item"]], log=log, start="approach")
                    return inner[0](t)
                goto("stow", t)
            else:
                st["lifted"], st["tries"] = t, 0
                goto("face", t)
        nose = LIFT_TO + (CARRY_NOSE - LIFT_TO) * min(1.0, (t - st["lifted"]) / RAISE_S)
        s = {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": nose, "lt": 1.0, "rt": 0.0, "hat_x": -1, "hat_y": 0, "buttons": frozenset()}
        base_p = robot.data.root_link_pos_w[0].cpu().numpy()
        yaw = _yaw(robot.data.root_link_quat_w[0].cpu().numpy())
        c, sn = math.cos(yaw), math.sin(yaw)
        phase, since = st["phase"], t - st["since"]
        name = names[st["item"]]
        item = item_pos(name)
        goal = plan["goals"][name]
        err = np.array(goal) - item[:2]                    # where the item has to go, in the body frame
        ex, ey = c * err[0] + sn * err[1], -sn * err[0] + c * err[1]

        if phase == "face":
            # The pickup is always made facing +x (fetch.py's approach); a drop made
            # facing elsewhere turns first, with the item up.
            heading(yaw, s)
            if since > 0.5 and st["turn"] == 0 and abs(_wrap(want_yaw - yaw)) < 0.12:
                st["back_from"] = base_p[:2].copy()
                goto("back", t)
        elif phase == "back":
            if float(np.linalg.norm(base_p[:2] - st["back_from"])) > MAX_BACK:
                log(f"\n[cleanup] walked back {MAX_BACK} m without lining {name} up with the bin; dropping it here")
                goto("open", t)
            elif ex < -ON_TARGET:
                s["ly"] = stick(base_p, t)                 # Ly positive is back
                heading(yaw, s)
            elif ex > ON_TARGET:
                s["ly"] = -stick(base_p, t)
            else:
                goto("across", t)
        elif phase == "across":
            if abs(ey) > ON_TARGET:
                s["lx"] = -stick(base_p, t) * (1 if ey > 0 else -1)   # Lx positive is right
                heading(yaw, s)
            else:
                goto("hover", t)
        elif phase == "hover":
            heading(yaw, s)
            if since > 1.0 and st["turn"] == 0:
                log(f"\n[cleanup] over the bin: {name} at ({item[0]:+.3f},{item[1]:+.3f},{item[2]:.3f}), "
                    f"{ex * 100:+.1f} / {ey * 100:+.1f} cm off its place")
                if (abs(ex) > ON_TARGET or abs(ey) > ON_TARGET) and st["tries"] < 2:
                    st["tries"] += 1                       # swung off while settling: line up again
                    goto("back", t)
                else:
                    goto("set_down", t)
        elif phase == "set_down":
            s["ry"] = CARRY_NOSE + (SET_NOSE - CARRY_NOSE) * min(1.0, since / SET_S)
            landed = item[2] < M_HALF_H + 0.012
            if (landed and since > 0.5) or since > SET_MAX_S:
                log(f"\n[cleanup] {name} {'on the floor' if landed else 'still up'} at z {item[2]:.3f}")
                goto("open", t)
        elif phase == "open":
            s["ry"], s["lt"] = SET_NOSE, 0.0
            if since > 1.0:
                goto("slip", t)
        elif phase == "slip":
            s["ry"], s["lt"] = SET_NOSE, 0.0
            s["rx"] = SLIP_TWIST * min(1.0, since / 0.5)
            if since > SLIP_S:
                log(f"\n[cleanup] let go of {name}: {'IN PLACE' if placed(item, goal) else 'MISSED its place'} "
                    f"at ({item[0]:+.3f},{item[1]:+.3f},{item[2]:.3f})")
                goto("retreat", t)
        elif phase == "retreat":
            s["ry"], s["lt"] = 0.0, 0.0
            s["ly"] = WALK
            if since > RETREAT_S:
                goto("stow", t)
        elif phase == "stow":
            s["lt"], s["hat_x"], s["ry"] = 0.0, 0, 0.0
            if since > STOW_S:
                if st["item"] + 1 < len(names):
                    st["item"], st["retries"] = st["item"] + 1, 0
                    inner[0] = fetch_pilot(env, player, prop=names[st["item"]], log=log, start="approach")
                    goto("fetch", t)
                else:
                    goto("celebrate" if plan["celebrate"] else "done", t)
        elif phase == "celebrate":
            s["lt"], s["hat_x"], s["ry"] = 0.0, 0, 0.0
            if since < 5.0:
                s["buttons"] = frozenset({"R3"})
                s["rx"] = 0.8 * math.sin(since * 2 * math.pi / 1.6)
            else:
                goto("done", t)
        elif phase == "done":
            s["lt"], s["hat_x"], s["ry"] = 0.0, 0, 0.0

        if t - last_log[0] >= 0.5:
            last_log[0] = t
            where = " ".join(f"{n}=({p[0]:+.3f},{p[1]:+.3f},{p[2]:.3f}{',placed' if placed(p, plan['goals'][n]) else ''})"
                             for n, p in ((n, item_pos(n)) for n in names))
            log(f"\n[cleanup] t={t:6.2f} {phase:<9} base=({base_p[0]:+.3f},{base_p[1]:+.3f}) "
                f"yaw={math.degrees(yaw):+5.1f} {where}")
        return GamepadState(**s)

    pilot.state = st    # read by a caller that runs the cleanup as one skill (tools/fetch/voice_pilot.py)
    return pilot
