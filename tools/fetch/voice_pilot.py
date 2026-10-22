"""A skill server: the crab does what it is told over a local socket.

    FETCH_LIFT=1 .venv/bin/mjpython tools/fetch/run.py --pilot tools/fetch/voice_pilot.py \\
        --task jumper.five_foot --cleanup --scene ../maps/home.map

Something else -- a voice front end, a script, `nc` -- connects to 127.0.0.1:$CRAB_PORT
(8765 by default) and sends one JSON object per line:

    {"skill": "walk", "distance": 0.5, "direction": "forward"}   forward/back/left/right, m
    {"skill": "turn", "degrees": 90}                               positive is left
    {"skill": "go_to", "x": 0.5, "y": -0.3}                         world metres: turn, then walk
    {"skill": "gesture", "name": "hello"}                           see GESTURES
    {"skill": "dance", "name": "crab"}                              see DANCES
    {"skill": "face", "degrees": 180}                               turn to a world heading
    {"skill": "tidy_up"}                                            cleanup.py's two items into the bin
    {"skill": "fetch_slippers"}                                     cleanup.py's slippers job: to the bedside
    {"skill": "follow", "seconds": 30, "distance": 0.6}             keep near the person (story.py)
    {"skill": "person", "pose": "standing"} / {"skill": "person", "walk_to": [x, y]}
                                                                    a stage direction for the person:
                                                                    done at once, beside whatever is queued
    {"skill": "stop"}                                               drop the queue and stand
    {"skill": "status"}                                             answered at once, nothing queued

Every line is answered with one JSON line. Skills queue and run one after another
in the controller's own loop; the socket thread only enqueues, so nothing it does
can race the simulation. Gestures and dances are the bundle's own modes, entered
by the pad combination the bundle's manual gives them, and a skill ends when the
controller hands back to locomotion. Ground truth stands in for odometry, as in
`fetch.py`.
"""
from __future__ import annotations

import json
import math
import os
import queue
import socketserver
import threading

import numpy as np
from fetch import TURN, TWIST_SIGN, _wrap, _yaw

PORT = int(os.environ.get("CRAB_PORT", "8765"))

#: Pad combinations, from out/bundle_example/jumper/manual.en.json. D-pad up is hat_y -1
#: (the kernel's sign: positive is down).
GESTURES = {
    "hello": {"hat_y": -1}, "bow": {"hat_y": 1}, "paw": {"hat_x": -1}, "salute": {"hat_x": 1},
    "sleep": {"buttons": {"Y"}}, "zoidberg": {"buttons": {"X"}}, "jump": {"buttons": {"A"}},
    "threat": {"buttons": {"menu", "X"}}, "pump": {"buttons": {"menu", "Y"}}, "wave": {"buttons": {"menu", "A"}},
}
DANCES = {
    "crab": {"buttons": {"menu"}, "hat_y": -1}, "brazilian": {"buttons": {"menu"}, "hat_y": 1},
    "maze": {"buttons": {"menu"}, "hat_x": -1}, "dream_wings": {"buttons": {"menu"}, "hat_x": 1},
}
DIRECTIONS = {"forward": (1.0, 0.0), "back": (-1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}
SKILLS = ("walk", "turn", "face", "go_to", "gesture", "dance", "tidy_up", "fetch_slippers", "follow",
          "person", "stop")
JOB_OF = {"tidy_up": "tidy", "fetch_slippers": "slippers"}
FOLLOW_GAP = 0.6                 # m from the person the crab stands at
FOLLOW_SLACK = 0.25              # it sets off again once the person is this much further
FOLLOW_CLOSE = 0.40              # and backs away, facing them, once they come this close (story_v2: the
                                 # person walked onto the crab, which only ever closed the gap)

WALK = 0.5                       # stick travel: above the policy's ~0.4 dead band
PRESS_S = 0.3                    # how long a mode switch is held
MODE_TIMEOUT_S = 40.0            # a gesture or dance that never hands back
MOVE_TIMEOUT_S = 25.0


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def make_pilot(env, player, log=print):
    from controller import GamepadState

    robot = env.scene["robot"]
    inbox: queue.Queue = queue.Queue()
    st = {"current": None, "since": 0.0, "pending": [], "done": [], "turn": 0,
          "stall_p": None, "stall_t": 0.0, "bump": 0.0, "status": {}}
    lock = threading.Lock()

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            for raw in self.rfile:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    cmd = json.loads(line)
                    skill = cmd.get("skill")
                    if skill == "status":
                        with lock:
                            reply = dict(st["status"])
                    elif skill in SKILLS:
                        _check(cmd)
                        inbox.put(cmd)
                        reply = {"ok": True, "queued": skill}
                    else:
                        reply = {"ok": False, "error": f"unknown skill {skill!r}"}
                except (ValueError, KeyError, TypeError) as error:
                    reply = {"ok": False, "error": str(error)}
                self.wfile.write((json.dumps(reply, ensure_ascii=False) + "\n").encode("utf-8"))

    def _check(cmd):
        skill = cmd["skill"]
        if skill == "walk":
            float(cmd["distance"])
            if cmd.get("direction", "forward") not in DIRECTIONS:
                raise ValueError(f"direction must be one of {sorted(DIRECTIONS)}")
        elif skill in ("turn", "face"):
            float(cmd["degrees"])
        elif skill == "go_to":
            float(cmd["x"]), float(cmd["y"])
        elif skill == "gesture" and cmd.get("name") not in GESTURES:
            raise ValueError(f"gesture must be one of {sorted(GESTURES)}")
        elif skill == "dance" and cmd.get("name") not in DANCES:
            raise ValueError(f"dance must be one of {sorted(DANCES)}")

    person = None
    if "person" in env.scene.entities:
        from story import Person

        person = Person(env)

    server = _Server(("127.0.0.1", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log(f"[voice] listening on 127.0.0.1:{PORT}")

    def pose():
        p = robot.data.root_link_pos_w[0].cpu().numpy()
        return p, _yaw(robot.data.root_link_quat_w[0].cpu().numpy())

    def stick(base_p, t):
        """WALK, raised while the base is not moving, as fetch.py's stick_mag does."""
        if st["stall_p"] is None or np.linalg.norm(base_p[:2] - st["stall_p"]) > 0.015:
            st["stall_p"], st["stall_t"] = base_p[:2].copy(), t
        elif t - st["stall_t"] > 1.0:
            st["bump"] = min(0.3, st["bump"] + 0.05)
            st["stall_t"] = t
        return WALK + st["bump"]

    def turn_toward(yaw, want, s, start=0.08, stop=0.05):
        """fetch.py's turn pulses; True once within `stop` of `want`."""
        e = _wrap(want - yaw)
        if st["turn"] == 0 and abs(e) > start:
            st["turn"] = 1 if e > 0 else -1
        elif st["turn"] != 0 and (abs(e) < stop or (e > 0) != (st["turn"] > 0)):
            st["turn"] = 0
        if st["turn"] != 0:
            s["rx"] = TURN * st["turn"] * TWIST_SIGN
        return st["turn"] == 0 and abs(e) <= start

    def begin(cmd, t, base_p, yaw):
        cmd = dict(cmd)
        if cmd["skill"] == "walk":
            fx, fy = DIRECTIONS[cmd.get("direction", "forward")]
            c, sn = math.cos(yaw), math.sin(yaw)
            d = abs(float(cmd["distance"]))
            cmd["_goal"] = base_p[:2] + d * np.array([c * fx - sn * fy, sn * fx + c * fy])
            cmd["_heading"] = yaw
            cmd["_axis"] = (fx, fy)
        elif cmd["skill"] == "turn":
            cmd["_want"] = _wrap(yaw + math.radians(float(cmd["degrees"])))
        elif cmd["skill"] == "face":
            cmd["_want"] = _wrap(math.radians(float(cmd["degrees"])))
        elif cmd["skill"] == "follow":
            cmd["_until"] = t + float(cmd.get("seconds", 30.0))
            cmd["_gap"] = float(cmd.get("distance", FOLLOW_GAP))
            cmd["_moving"] = False
        elif cmd["skill"] == "go_to":
            cmd["_goal"] = np.array([float(cmd["x"]), float(cmd["y"])])
            cmd["_stage"] = "turn"
        elif cmd["skill"] in ("gesture", "dance"):
            table = GESTURES if cmd["skill"] == "gesture" else DANCES
            cmd["_press"] = table[cmd["name"]]
            cmd["_entered"] = False
        elif cmd["skill"] in JOB_OF:
            from cleanup import make_pilot as cleanup_pilot

            cmd["_inner"] = cleanup_pilot(env, player, log=log, job=JOB_OF[cmd["skill"]])
            cmd["_stage"] = "run"
        st.update(current=cmd, since=t, turn=0, stall_p=None, bump=0.0)
        log(f"\n[voice] {t:6.2f}s  start {_label(cmd)}")

    def finish(t, outcome="done"):
        cmd = st["current"]
        p, yaw = pose()
        log(f"\n[voice] {t:6.2f}s  {outcome} {_label(cmd)}  at ({p[0]:+.3f},{p[1]:+.3f}) "
            f"heading {math.degrees(yaw):+.0f} deg")
        st["done"].append({"skill": _label(cmd), "outcome": outcome, "t": round(t, 2)})
        st["current"] = None

    def pilot(t):
        s = dict(lx=0.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0, hat_x=0, hat_y=0, buttons=frozenset())
        base_p, yaw = pose()
        fsm = player.fsm
        while not inbox.empty():
            cmd = inbox.get_nowait()
            if cmd["skill"] == "person":
                if person is not None:
                    person.command(cmd)
                    log(f"\n[voice] {t:6.2f}s  person {_label(cmd)}")
            elif cmd["skill"] == "stop":
                st["pending"].clear()
                if st["current"] is not None:
                    finish(t, "stopped")
            else:
                st["pending"].append(cmd)
        if st["current"] is None and st["pending"] and fsm.mode == "locomotion":
            begin(st["pending"].pop(0), t, base_p, yaw)
        cmd, since = st["current"], t - st["since"]
        if person is not None:
            person.step(env.step_dt)

        if cmd is None:
            pass
        elif cmd["skill"] == "walk":
            err = cmd["_goal"] - base_p[:2]
            c, sn = math.cos(yaw), math.sin(yaw)
            ex, ey = c * err[0] + sn * err[1], -sn * err[0] + c * err[1]
            fx, fy = cmd["_axis"]
            along = ex * fx + ey * fy
            if along < 0.03 or since > MOVE_TIMEOUT_S:
                finish(t, "done" if along < 0.03 else "timed out")
            else:
                mag = stick(base_p, t)
                if fx:
                    s["ly"] = -mag * fx                      # Ly positive is back
                else:
                    s["lx"] = -mag * fy                      # Lx positive is right
                turn_toward(yaw, cmd["_heading"], s, start=0.12, stop=0.04)
        elif cmd["skill"] == "follow":
            if person is None or t > cmd["_until"]:
                finish(t, "done" if person is not None else "no one to follow")
            else:
                err = person.xy - base_p[:2]
                dist = float(np.linalg.norm(err))
                bearing = math.atan2(err[1], err[0])
                if dist > cmd["_gap"] + FOLLOW_SLACK:
                    cmd["_moving"] = True
                elif dist < cmd["_gap"]:
                    cmd["_moving"] = False
                facing = turn_toward(yaw, bearing, s, start=0.25 if cmd["_moving"] else 0.35, stop=0.08)
                if cmd["_moving"] and (facing or abs(_wrap(bearing - yaw)) < 0.5):
                    s["ly"] = -stick(base_p, t)
                elif dist < FOLLOW_CLOSE and abs(_wrap(bearing - yaw)) < 0.8:
                    s["ly"] = stick(base_p, t)                # Ly positive is back
        elif cmd["skill"] in ("turn", "face"):
            if turn_toward(yaw, cmd["_want"], s) and since > 0.3 or since > MOVE_TIMEOUT_S:
                finish(t)
        elif cmd["skill"] == "go_to":
            err = cmd["_goal"] - base_p[:2]
            dist = float(np.linalg.norm(err))
            if dist < 0.05 or since > 2 * MOVE_TIMEOUT_S:
                finish(t, "done" if dist < 0.05 else "timed out")
            elif cmd["_stage"] == "turn":
                if turn_toward(yaw, math.atan2(err[1], err[0]), s, start=0.06, stop=0.03):
                    cmd["_stage"] = "walk"
            else:
                s["ly"] = -stick(base_p, t)
                turn_toward(yaw, math.atan2(err[1], err[0]), s, start=0.15, stop=0.05)
        elif cmd["skill"] in ("gesture", "dance"):
            if since < PRESS_S:
                p = cmd["_press"]
                s["hat_x"], s["hat_y"] = p.get("hat_x", 0), p.get("hat_y", 0)
                s["buttons"] = frozenset(p.get("buttons", ()))
            if fsm.mode != "locomotion":
                cmd["_entered"] = True
            elif cmd["_entered"] or since > MODE_TIMEOUT_S:
                finish(t, "done" if cmd["_entered"] else "never entered")
        elif cmd["skill"] in JOB_OF:
            if cmd["_stage"] == "run":
                if cmd["_inner"].state["phase"] != "done":
                    pad = cmd["_inner"](t)
                    _remember(st, lock, base_p, yaw, fsm, cmd, t, person)
                    return pad
                cmd["_stage"], cmd["_left_at"] = "leave", t
            if t - cmd["_left_at"] < PRESS_S:
                s["buttons"] = frozenset({"LB"})          # claw_left's own switch, back to locomotion
            elif fsm.mode == "locomotion":
                finish(t)

        _remember(st, lock, base_p, yaw, fsm, st["current"], t, person)
        return GamepadState(**s)

    return pilot


def _remember(st, lock, base_p, yaw, fsm, cmd, t, person) -> None:
    with lock:
        st["status"] = {
            "ok": True, "t": round(t, 2), "mode": fsm.mode,
            "x": round(float(base_p[0]), 3), "y": round(float(base_p[1]), 3),
            "heading_deg": round(math.degrees(yaw), 1),
            "doing": _label(cmd) if cmd else None,
            "queued": [_label(c) for c in st["pending"]],
            "recent": st["done"][-5:],
        }
        if person is not None:
            st["status"]["person"] = {"pose": person.pose, "x": round(float(person.pos[0]), 3),
                                      "y": round(float(person.pos[1]), 3), "walking": person.walking}


def _label(cmd) -> str:
    if cmd is None:
        return "nothing"
    extra = {k: v for k, v in cmd.items() if k != "skill" and not k.startswith("_")}
    return cmd["skill"] + (" " + json.dumps(extra, ensure_ascii=False) if extra else "")
