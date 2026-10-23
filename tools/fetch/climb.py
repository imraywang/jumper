"""The climb film: three steps up the side of home.map's bed, a hop each, then a tap on the sleeper.

    .venv/bin/python tools/fetch/run.py --pilot tools/fetch/climb.py --task jumper.five_foot \\
        --climb --scene ../maps/home.map --app ../out/bundle_film/jumper.app --headless --steps 30000

The steps are a pouf, an ottoman and a bench, each STEP_H taller than the last, in a
row along the bed's open side so the last is level with the mattress. They collide;
the crab gets up them with the bundle's dash mode, which in the film's bundle is
`jumper.step_up` (a forward hop trained to land on a step of up to 0.15 m) under the
dash's name. On the mattress it walks up beside the sleeper and taps with the paw
gesture.

The person is `story.py`'s, drawn only. home.map's blanket is made drawn only too: it
is a 4 cm ledge on top of the mattress where the crab comes on (and the pillow, which its
claws come close to, likewise), and the film's bed
(blender/furnish.py) folds its duvet at the foot, clear of the crab's way.

The pilot is a script of beats (`BEATS`), each logged with its simulation time so the
edit can put the lines on them.
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

#: Mattress top of home.map's bed (bed_mattress, z 0.27..0.42).
BED_TOP = 0.42
#: How many steps share BED_TOP: 3 x 0.14 m, or 4 x 0.105 m if the hop is surer lower down.
N_STEPS = int(os.environ.get("CLIMB_N", "3"))
#: Depth of each step along y (m). At 0.45 the hop came down with the claws at the next step's
#: face and the landing pitched the crab into it (climb_t2/t3, 2026-10-22); 0.6 leaves room to stop.
DEPTH = 0.6
NAMES = {3: ("pouf", "ottoman", "bench"), 4: ("pouf", "stool", "ottoman", "bench")}[N_STEPS]
LAST_Y = 1.30                     # the bench's far end, short of the nightstand; it runs past the sleeper's chest
#: (name, y from, y to, top): the steps' extent along y; all span STEP_X in x.
STEPS = tuple((n, LAST_Y - (N_STEPS - i) * DEPTH, LAST_Y - (N_STEPS - i - 1) * DEPTH, BED_TOP * (i + 1) / N_STEPS)
              for i, n in enumerate(NAMES))
STEP_X = (-0.94, -0.34)
#: Base to the next step's face at the hop, metres: the near end of what step_up trained on (0.21-0.25).
HOP_GAP = 0.21
LANE_X = sum(STEP_X) / 2
START = (0.25, STEPS[0][1] - HOP_GAP, math.pi)
#: Free props of home.map that sit where the crab walks: moved out of the way at the start.
MOVE_AWAY = {"plate": (-1.55, -1.75), "book_flat_bedside": (-1.72, -1.58)}

TURN = 0.75
TWIST_SIGN = -1.0
WALK = 0.5
PRESS_S = 0.3

UP = math.pi / 2
#: Where the crab stands on the mattress to tap: base 0.19 m behind its claws, facing -x, the
#: claws at the sleeper's upper arm (story.py's person lies centred on x -1.37) and its
#: side legs (0.2 m out) clear of the bench's near end at y 0.70 -- at 0.85 they hung over the
#: ottoman and the crab sat tilted 20 deg and could not walk (climb_t9).
TAP_X, TAP_Y = -1.00, 0.95
DEBUG = bool(os.environ.get("CLIMB_DEBUG"))
TO_BED = math.pi


def _face_y(name):
    return next(s[1] for s in STEPS if s[0] == name)


BEATS = (
    ("mark", "night"),
    ("wait", 2.0),
    ("gesture", "salute"),              # goodnight
    ("wait", 3.0),
    ("mark", "morning"),
    ("goto", LANE_X, _face_y(NAMES[0]) - HOP_GAP, TO_BED),        # to the bedside: too high
    ("face", UP),                       # looks along the bed
    ("wait", 1.0),
    *[b for k, n in enumerate(NAMES) for b in (("goto", LANE_X, _face_y(n) - HOP_GAP, UP),
                                               ("mark", f"hop{k + 1}"), ("hop", n))],
    ("goto", LANE_X, TAP_Y, UP),
    ("face", TO_BED),
    ("mark", "onto_bed"),
    ("goto", TAP_X, TAP_Y, TO_BED),      # front claws at the sleeper's arm
    ("mark", "tap"),
    ("gesture", "paw"),
    ("gesture", "paw"),
    ("mark", "awake"),
    ("wait", 1.5),
    ("gesture", "hello"),
    ("wait", 3.0),
    ("mark", "end"),
)

#: Where a climb starts over after a fall: the walk to the foot of the first step.
RESTART = next(i for i, b in enumerate(BEATS) if b[0] == "goto")

GESTURES = {"hello": {"hat_y": -1}, "bow": {"hat_y": 1}, "paw": {"hat_x": -1}, "salute": {"hat_x": 1}}


def _decorate(spec: Any) -> None:
    import mujoco

    rgba = {"pouf": [0.80, 0.62, 0.45, 1], "stool": [0.70, 0.55, 0.40, 1], "ottoman": [0.45, 0.52, 0.48, 1],
            "bench": [0.55, 0.40, 0.28, 1]}
    for name, y0, y1, top in STEPS:
        spec.worldbody.add_geom(name=f"climb_{name}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=[(STEP_X[1] - STEP_X[0]) / 2, (y1 - y0) / 2, top / 2],
                                pos=[LANE_X, (y0 + y1) / 2, top / 2], rgba=rgba[name], group=2, mass=0.0)
    for g in spec.geoms:
        if g.name.endswith("bed_blanket") or g.name.endswith("bed_pillow"):
            g.contype = g.conaffinity = 0


def apply_climb(env_cfg: Any) -> None:
    from mjlab.entity import EntityCfg
    from story import LIE_MAT, LIE_POS, _quat, person_spec

    from scenes.registry import _install_spec_edits
    from tasks.jumper.five_foot import objects as O

    env_cfg.sim.nconmax = max(int(env_cfg.sim.nconmax or 0), 1024)
    env_cfg.sim.njmax = max(int(env_cfg.sim.njmax or 0), 8192)
    _install_spec_edits(env_cfg.scene, None, _decorate)
    props = {"person": EntityCfg(spec_fn=person_spec, init_state=EntityCfg.InitialStateCfg(
        pos=LIE_POS, rot=tuple(_quat(LIE_MAT))))}
    env_cfg.scene.entities.update(props)
    O._place_props(env_cfg, props)


def _yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _move_free_props(env, log):
    import mujoco
    import torch

    m = env.sim.mj_model
    qpos, qvel = env.sim.data.qpos, env.sim.data.qvel
    origin = env.scene.env_origins[0].cpu().numpy()
    n = 0
    for j in range(m.njnt):
        name = m.joint(j).name
        to = next((xy for key, xy in MOVE_AWAY.items() if key in name), None)
        if to is None or m.jnt_type[j] != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        a, v = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        qpos[0, a:a + 7] = torch.tensor([to[0] + origin[0], to[1] + origin[1], 0.02 + 0.035 * n, 1, 0, 0, 0],
                                        dtype=qpos.dtype, device=qpos.device)
        qvel[0, v:v + 6] = 0.0
        n += 1
    log(f"[climb] moved {n} props out of the crab's way")


def make_pilot(env, player, log=print):
    import torch
    from story import Person

    from controller import GamepadState

    robot = env.scene["robot"]
    origin = env.scene.env_origins[0].cpu().numpy()
    foot_ids = [i for i, n in enumerate(robot.body_names) if n.endswith(("foot_tip_link", "palm_pad_b_link"))]
    person = Person(env)
    _move_free_props(env, log)
    st = {"i": 0, "since": None, "turn": 0, "stall_p": None, "stall_t": 0.0, "bump": 0.0, "phase": "",
          "placed": False, "marks": {}}

    def where():
        p = robot.data.root_link_pos_w[0].cpu().numpy() - origin
        return p, _yaw(robot.data.root_link_quat_w[0].cpu().numpy())

    def face(yaw, want, s):
        e = _wrap(want - yaw)
        if st["turn"] == 0 and abs(e) > 0.06:
            st["turn"] = 1 if e > 0 else -1
        elif st["turn"] != 0 and (abs(e) < 0.02 or (e > 0) != (st["turn"] > 0)):
            st["turn"] = 0
        if st["turn"]:
            s["rx"] = TURN * st["turn"] * TWIST_SIGN
        return abs(e) < 0.06

    def stick(p, t):
        if st["stall_p"] is None or np.linalg.norm(p[:2] - st["stall_p"]) > 0.015:
            st["stall_p"], st["stall_t"] = p[:2].copy(), t
        elif t - st["stall_t"] > 1.0:
            st["bump"], st["stall_t"] = min(0.3, st["bump"] + 0.05), t
        return WALK + st["bump"]

    def walk(p, yaw, goal, s, t, tol=0.02):
        """fetch.py's walk_to: strafe onto the line, then along it, holding the heading."""
        err = np.asarray(goal) - p[:2]
        c, sn = math.cos(yaw), math.sin(yaw)
        ex, ey = c * err[0] + sn * err[1], -sn * err[0] + c * err[1]
        mag = stick(p, t)
        if abs(ey) > 0.02:
            s["lx"] = -mag * (1 if ey > 0 else -1)
            return False
        if abs(ex) > tol:
            s["ly"] = -mag * (1 if ex > 0 else -1)
            s["lx"] = -max(-0.2, min(0.2, 8.0 * ey))
            return False
        return True

    def nxt(t):
        st.update(i=st["i"] + 1, since=t, turn=0, stall_p=None, bump=0.0, phase="")

    def pilot(t):
        s = {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0, "lt": 0.0, "rt": 0.0, "hat_x": 0, "hat_y": 0, "buttons": frozenset()}
        person.step(env.step_dt)
        if not st["placed"]:
            # Spend the night where the film opens: written once, on the first tick.
            x, y, yaw = START
            q = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
            root = torch.tensor([[x + origin[0], y + origin[1], 0.11, *q, 0, 0, 0, 0, 0, 0]],
                                dtype=torch.float32, device=env.device)
            robot.write_root_state_to_sim(root)
            st.update(placed=True, since=t)
            return GamepadState(**s)
        if st["i"] >= len(BEATS):
            return GamepadState(**s)
        p, yaw = where()
        # The app's fall handler puts the robot back at the map's spawn: start the climb over
        # from the first step, by the way round the end of the row (the goto after "morning"
        # strafes onto the first step's line before walking along it, clear of the steps' side).
        if st.get("last_p") is not None and np.linalg.norm(p[:2] - st["last_p"]) > 0.3:
            st["resets"] = st.get("resets", 0) + 1
            log(f"\n[climb] {t:7.2f}s  reset to ({p[0]:+.3f},{p[1]:+.3f}) -- climbing again (reset {st['resets']})")
            # The map's spawn area overlaps the steps: the robot came back wedged against the
            # stool at 40 deg (climb_t6). Back to where the film starts, on the next tick.
            st.update(i=RESTART, since=t, phase="", turn=0, stall_p=None, bump=0.0, placed=False)
        st["last_p"] = p[:2].copy()
        beat, since = BEATS[st["i"]], t - st["since"]
        mode = player.fsm.mode
        if DEBUG and mode != st.get("last_mode"):
            log(f"[climb] mode {t:6.2f} {st.get('last_mode')} -> {mode}  beat {beat}")
            st["last_mode"] = mode
        kind = beat[0]
        if kind == "mark":
            st["marks"][beat[1]] = round(t, 2)
            log(f"\n[climb] {t:7.2f}s  {beat[1]}  base ({p[0]:+.3f},{p[1]:+.3f},{p[2]:.3f}) yaw {math.degrees(yaw):+.0f}")
            nxt(t)
        elif kind == "wait":
            if since > beat[1]:
                nxt(t)
        elif kind == "face":
            if face(yaw, beat[1], s) and since > 0.4:
                nxt(t)
        elif kind == "goto":
            _, x, y, want = beat
            if st["phase"] == "":
                st["phase"] = "aim"
            if st["phase"] == "aim":
                if face(yaw, want, s):
                    st["phase"] = "walk"
            elif st["phase"] == "walk":
                if abs(_wrap(want - yaw)) > 0.12:
                    st["phase"] = "aim"
                elif walk(p, yaw, (x, y), s, t):
                    st["phase"], st["since"] = "settle", t
            elif st["phase"] == "settle":
                ok = face(yaw, want, s)
                if t - st["since"] > 0.8 and ok:
                    nxt(t)
        elif kind == "gesture":
            if since < PRESS_S:
                g = GESTURES[beat[1]]
                s["hat_x"], s["hat_y"] = g.get("hat_x", 0), g.get("hat_y", 0)
                s["buttons"] = frozenset(g.get("buttons", ()))
            elif since > 1.0 and mode == "locomotion":
                # Timed on its own clock: restarting `since` here re-pressed the button within
                # PRESS_S and the gesture played twice (climb_t7's salute, and the hop below).
                if st["phase"] == "":
                    st["phase"], st["t_back"] = "back", t
                elif t - st["t_back"] > 0.5:
                    nxt(t)
        elif kind == "hop":
            top = next(s_[3] for s_ in STEPS if s_[0] == beat[1])
            if since < PRESS_S:
                s["buttons"] = frozenset({"L3"})
            elif st["phase"] == "" and mode == "dash":
                st["phase"] = "in"
            elif st["phase"] == "in" and mode == "locomotion":
                st["phase"], st["t_land"] = "landed", t
            elif st["phase"] == "landed" and t - st["t_land"] > 1.2:
                # Every foot over the step, not only the base high enough: perched with the rear
                # legs over the edge, the crab passed a height check and fell back (climb_t6).
                face_y = next(s_[1] for s_ in STEPS if s_[0] == beat[1])
                feet = robot.data.body_link_pos_w[0, foot_ids].cpu().numpy() - origin
                # The hop carries the base ~0.35 m in the app and the rear tips can come down a few
                # cm short of the edge (climb_t8); the walk on to the next hop brings them up.
                on = p[2] > top + 0.06 and p[1] > face_y + 0.10 and bool((feet[:, 1] > face_y - 0.06).all())
                log(f"\n[climb] {t:7.2f}s  hop onto the {beat[1]}: base ({p[0]:+.3f},{p[1]:+.3f},{p[2]:.3f}) "
                    f"{'ON' if on else 'MISSED'}")
                if on:
                    nxt(t)
                else:
                    # Back to where the hop starts and try again (the goto before it).
                    st.update(i=st["i"] - 2, since=t, phase="", turn=0)
            elif since > 8.0 and st["phase"] == "":
                st["since"] = t           # the press did not take: press again
        if DEBUG and int(t * 4) != st.get("pl"):
            st["pl"] = int(t * 4)
            g = robot.data.projected_gravity_b[0].cpu().numpy()
            log(f"[climb] dbg {t:6.2f} {kind}:{st['phase']} mode {mode} base ({p[0]:+.3f},{p[1]:+.3f},{p[2]:.3f}) "
                f"yaw {math.degrees(yaw):+.0f} tilt {math.degrees(math.acos(max(-1, min(1, -g[2])))):.0f} "
                f"stick ({s['lx']:+.2f},{s['ly']:+.2f},{s['rx']:+.2f})")
        return GamepadState(**s)

    pilot.st = st
    return pilot
