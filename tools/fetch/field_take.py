"""The training field as a take for blender/: many envs of jumper.step_up hopping at once.

    .venv/bin/python tools/fetch/field_take.py --checkpoint <model.pt> --out ../runs/field_take

`export_poses.PoseExport` records one environment, which is all a story needs. The
training field is the point of this shot: COLS terrain columns by step_up's ten rows of
step height, one robot on every patch, each hopping when its command's delay runs out
(widened here so that they do not all go at once). The take is written in
export_poses' format with every robot after the first as extra bodies -- `robot#k/...`,
a copy of the robot's geoms on each -- so `blender/import_take.py` builds it unchanged.
"""
from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

COLS = 20                     # terrain columns for the shot (step_up trains on 5)
SECONDS = 7.0
FPS = 30.0
DELAY = (0.3, 3.5)            # s before each robot hops: a field that goes off unevenly


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cols", type=int, default=COLS)
    ap.add_argument("--seconds", type=float, default=SECONDS)
    args = ap.parse_args()

    from mjrl.backend.select import use_backend

    import tasks

    R = importlib.import_module("mjrl.backend.resolve")
    cfg = tasks.load_env_cfg("jumper.step_up", play=True)
    gen = cfg.scene.terrain.terrain_generator
    first = next(iter(gen.sub_terrains.values()))
    from tasks.jumper.step_up.env_cfg import FACES
    from tasks.jumper.step_up.mdp.terrain import StepTerrainCfg

    gen.sub_terrains = {f"c{k:02d}": StepTerrainCfg(size=first.size, step_height_range=first.step_height_range,
                                                     face=FACES[k % len(FACES)]) for k in range(args.cols)}
    gen.num_cols = args.cols
    n = gen.num_rows * args.cols
    res = R.resolve(backend="native", device="cpu", num_envs=n)
    use_backend(res)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    cfg.scene.num_envs = n
    cfg.episode_length_s = args.seconds + 1.0
    cfg.commands["jump"].command_delay_range = DELAY
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    wrapped = RslRlVecEnvWrapper(env)
    agent = tasks.load_agent_cfg("jumper.step_up")
    agent.logger = "tensorboard"
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent), None, "cpu")
    runner.load(str(args.checkpoint))
    policy = runner.get_inference_policy(device="cpu")

    # One robot per patch: env i on row i % rows, column i // rows.
    terrain = env.scene.terrain
    rows = terrain.terrain_origins.shape[0]
    terrain.terrain_levels[:] = torch.arange(n) % rows
    terrain.terrain_types[:] = torch.arange(n) // rows
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    wrapped.reset()
    obs = wrapped.get_observations()

    from export_poses import dump_geoms

    out = args.out.resolve()
    model = getattr(env.sim, "render_model", env.sim.mj_model)
    dump_geoms(model, out)
    info = json.loads((out / "geoms.json").read_text())
    bodies = info["bodies"]
    robot = [b for b, name in enumerate(bodies) if name.startswith("robot/")]
    robot_geoms = [g for g in info["geoms"] if g["body"] in robot]
    remap = {}
    for k in range(1, n):
        for b in robot:
            remap[(k, b)] = len(bodies)
            bodies.append(bodies[b].replace("robot/", f"robot#{k}/", 1))
    for k in range(1, n):
        for g in robot_geoms:
            info["geoms"].append({**g, "body": remap[(k, g["body"])], "body_name": bodies[remap[(k, g["body"])]]})
    (out / "geoms.json").write_text(json.dumps(info, ensure_ascii=False))

    data = env.sim.data
    frames_xpos, frames_xquat = [], []
    steps = round(args.seconds / env.step_dt)
    every = 1.0 / FPS
    next_t, t = 0.0, 0.0
    with torch.inference_mode():
        for s in range(steps + 1):
            if t + 1e-9 >= next_t:
                xp = data.xpos.cpu().numpy()        # [n, nbody, 3]
                xq = data.xquat.cpu().numpy()
                row_p, row_q = [xp[0, i] for i in range(xp.shape[1])], [xq[0, i] for i in range(xq.shape[1])]
                for k in range(1, n):
                    for b in robot:
                        row_p.append(xp[k, b])
                        row_q.append(xq[k, b])
                frames_xpos.append(np.stack(row_p).astype(np.float32))
                frames_xquat.append(np.stack(row_q).astype(np.float32))
                next_t += every
            obs, _, _, _ = wrapped.step(policy(obs))
            t += env.step_dt
    nf = len(frames_xpos)
    np.savez_compressed(out / "frames.npz", t=np.arange(nf, dtype=np.float32) / FPS,
                        xpos=np.stack(frames_xpos), xquat=np.stack(frames_xquat), fps=FPS,
                        lookat=np.zeros((nf, 3), np.float32), distance=np.full(nf, 3.0, np.float32),
                        azimuth=np.full(nf, 90.0, np.float32), elevation=np.full(nf, -30.0, np.float32))
    from tasks.jumper.step_up.mdp.rewards import on_step
    ok = on_step(env)
    print(f"[field] {n} robots, {nf} frames to {out}; on the step at the end: "
          + " ".join(f"row{r}:{float(ok[terrain.terrain_levels == r].mean()):.2f}" for r in range(rows)))


if __name__ == "__main__":
    main()
