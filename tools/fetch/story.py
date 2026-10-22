"""The alarm film's room: home.map's bedroom corner, the props, and a person to wake.

    FETCH_LIFT=1 .venv/bin/mjpython tools/fetch/run.py --pilot tools/fetch/voice_pilot.py \\
        --task jumper.five_foot --story --scene ../maps/home.map

`apply_story` adds everything `cleanup.apply_cleanup` does (the bag, the mug, the
bin), the slippers off the rug's corner, and a person. The person is drawn only --
every geom has contype and conaffinity 0, so nothing the robot does can push it and
nothing it does can push the robot -- and it is moved, not simulated: `Person`
writes its pose every control step, from a pose name or a walk toward a point.

Sized at 0.85 of an adult so that, lying on home.map's bed (1.66 m long, the pillow
at y = 1.24), the head is on the pillow and the feet stay on the mattress.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

SCALE = 0.85
#: The person's frame: the origin between the feet on the floor, +x the way it faces,
#: +z up. Each part is (geom type, size, position, rgba), before SCALE.
SKIN, SHIRT, JEANS, HAIR = [0.93, 0.76, 0.64, 1], [0.95, 0.95, 0.92, 1], [0.25, 0.33, 0.52, 1], [0.12, 0.09, 0.08, 1]
PARTS = (
    ("capsule", (0.06, 0.38), (0.0, 0.09, 0.47), JEANS),            # legs: radius, half length
    ("capsule", (0.06, 0.38), (0.0, -0.09, 0.47), JEANS),
    ("box", (0.12, 0.05, 0.03), (0.05, 0.09, 0.03), SHIRT),         # feet
    ("box", (0.12, 0.05, 0.03), (0.05, -0.09, 0.03), SHIRT),
    ("capsule", (0.16, 0.22), (0.0, 0.0, 1.12), SHIRT),             # torso
    ("capsule", (0.045, 0.24), (0.0, 0.21, 1.12), SHIRT),           # arms
    ("capsule", (0.045, 0.24), (0.0, -0.21, 1.12), SHIRT),
    ("sphere", (0.11,), (0.0, 0.0, 1.58), SKIN),                    # head
    ("sphere", (0.105,), (-0.02, 0.0, 1.62), HAIR),
)

#: Lying on its back on the bed, head on the pillow: the body's +z along the world's
#: +y, its face (+x) up.
LIE_POS = (-1.37, 1.24 - 1.58 * SCALE, 0.52)
LIE_MAT = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
#: Standing up beside the bed, facing into the room, in the slippers.
STAND_POS = (-0.70, 0.28, 0.0)    # feet in the slippers, as cleanup.SLIPPERS_GOAL leaves them
GET_UP_S = 2.0                    # lying to standing, interpolated
WALK_SPEED = 0.8                  # m/s: a slow indoor walk, what the Mixamo walk cycle plays at 0.8 of its rate


def person_spec() -> Any:
    import mujoco

    from tasks.jumper.five_foot import objects as O

    spec = O._prop_spec()
    body = O._free_body(spec, "person")
    for i, (kind, size, pos, rgba) in enumerate(PARTS):
        gtype = {"capsule": mujoco.mjtGeom.mjGEOM_CAPSULE, "box": mujoco.mjtGeom.mjGEOM_BOX,
                 "sphere": mujoco.mjtGeom.mjGEOM_SPHERE}[kind]
        sz = [v * SCALE for v in size] + [0.0] * (3 - len(size))
        body.add_geom(name=f"person_{i}", type=gtype, size=sz, pos=[v * SCALE for v in pos], rgba=list(rgba),
                      contype=0, conaffinity=0, group=2, mass=0.01)
    return spec


def apply_story(env_cfg: Any) -> None:
    import cleanup as C
    import mug as M
    from mjlab.entity import EntityCfg
    from slippers import slippers_spec

    from tasks.jumper.five_foot import objects as O

    C.apply_cleanup(env_cfg)
    props = {
        "slippers": EntityCfg(spec_fn=slippers_spec, init_state=EntityCfg.InitialStateCfg(
            pos=(*C.SLIPPERS_XY, M.BODY_HALF_H), rot=(1.0, 0.0, 0.0, 0.0))),
        "person": EntityCfg(spec_fn=person_spec, init_state=EntityCfg.InitialStateCfg(
            pos=LIE_POS, rot=tuple(_quat(LIE_MAT)))),
    }
    env_cfg.scene.entities.update(props)
    O._place_props(env_cfg, props)


def _quat(mat: np.ndarray) -> np.ndarray:
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat, dtype=np.float64).ravel())
    return q


def _yaw_mat(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


#: Where home.map's bedside books are moved to before the film starts: a stack in the
#: room's far corner. Left at the bedside they sat between the bed and the wake-up spot,
#: the crab pushed them along as it left, and the slippers came down on them
#: (story_film: set down 8 cm up, 13 cm off their place).
BOOKS_TO = (-1.72, -1.58)


def clear_bedside(env, log=print) -> None:
    """Move home.map's bedside books (free bodies of the imported world) out of the way."""
    import torch

    m = env.sim.mj_model
    qpos, qvel = env.sim.data.qpos, env.sim.data.qvel
    origin = env.scene.env_origins[0].cpu().numpy()
    moved = 0
    for j in range(m.njnt):
        if "book_flat_bedside" in m.joint(j).name and m.jnt_type[j] == mujoco_free():
            a, v = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
            z = 0.017 + 0.033 * moved
            qpos[0, a:a + 7] = torch.tensor([BOOKS_TO[0] + origin[0], BOOKS_TO[1] + origin[1], z, 1, 0, 0, 0],
                                            dtype=qpos.dtype, device=qpos.device)
            qvel[0, v:v + 6] = 0.0
            moved += 1
    log(f"[story] moved {moved} bedside books to {BOOKS_TO}")


def mujoco_free() -> int:
    import mujoco

    return int(mujoco.mjtJoint.mjJNT_FREE)


class Person:
    """Where the person is and what it is doing; `step` writes it to the simulation."""

    def __init__(self, env):
        self._env = env
        self.pose = "lying"
        self._t = 0.0
        self._from = None
        self.pos = np.array(LIE_POS, dtype=float)
        self.mat = LIE_MAT.copy()
        self.yaw = 0.0
        self._goal = None
        clear_bedside(env)

    @property
    def walking(self) -> bool:
        return self.pose == "getting_up" or self._goal is not None

    @property
    def xy(self) -> np.ndarray:
        return self.pos[:2].copy()

    def command(self, cmd: dict) -> None:
        """{"pose": "lying" | "standing"} or {"walk_to": [x, y]}."""
        if cmd.get("pose") == "standing" and self.pose == "lying":
            self.pose, self._t, self._from = "getting_up", 0.0, (self.pos.copy(), self.mat.copy())
        elif cmd.get("pose") == "lying":
            self.pose, self.pos, self.mat, self._goal = "lying", np.array(LIE_POS, float), LIE_MAT.copy(), None
        if cmd.get("walk_to") is not None and self.pose == "standing":
            self._goal = np.array(cmd["walk_to"], dtype=float)

    def step(self, dt: float) -> None:
        import torch

        self._t += dt
        if self.pose == "getting_up":
            k = min(1.0, self._t / GET_UP_S)
            k = k * k * (3 - 2 * k)
            p0, m0 = self._from
            self.yaw = 0.0
            target = _yaw_mat(self.yaw)
            # Slerp by way of the rotation between the two, about its own axis.
            q0, q1 = _quat(m0), _quat(target)
            q = _slerp(q0, q1, k)
            self.pos = (1 - k) * p0 + k * np.array(STAND_POS)
            self.pos[2] = (1 - k) * p0[2] + k * STAND_POS[2] + 0.25 * math.sin(math.pi * k)
            self.mat = _mat(q)
            if k >= 1.0:
                self.pose = "standing"
        elif self.pose == "standing" and self._goal is not None:
            d = self._goal - self.pos[:2]
            dist = float(np.linalg.norm(d))
            if dist < 0.03:
                self._goal = None
                self.pos[2] = 0.0
            else:
                want = math.atan2(d[1], d[0])
                err = (want - self.yaw + math.pi) % (2 * math.pi) - math.pi
                self.yaw += max(-3.0 * dt, min(3.0 * dt, err))
                if abs(err) < 0.6:
                    self.pos[:2] += d / dist * min(dist, WALK_SPEED * dt)
                self.pos[2] = 0.015 * abs(math.sin(self._t * 2 * math.pi * 1.8))   # a step's bob
            self.mat = _yaw_mat(self.yaw)
        entity = self._env.scene["person"]
        dev = self._env.device
        origin = self._env.scene.env_origins[0].cpu().numpy()
        pose = np.concatenate([self.pos + origin, _quat(self.mat)])
        entity.write_root_link_pose_to_sim(torch.tensor(pose[None], dtype=torch.float32, device=dev))
        entity.write_root_com_velocity_to_sim(torch.zeros(1, 6, device=dev))


def _slerp(q0: np.ndarray, q1: np.ndarray, k: float) -> np.ndarray:
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + k * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - k) * th) * q0 + math.sin(k * th) * q1) / math.sin(th)


def _mat(q: np.ndarray) -> np.ndarray:
    import mujoco

    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)
