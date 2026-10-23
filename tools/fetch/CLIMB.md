# The climb: a hop onto a step, four of them up the side of a bed

A film made for the 螃蟹机器人开源动作大赛 (Crab Robot Open-Source Motion Contest): in
home.map's bedroom the crab hops up four steps beside the bed, walks onto the mattress
and taps the sleeper awake. Everything below is **simulation** (MuJoCo, native backend,
an M4 Pro Mac); nothing here has run on a robot yet.

It is built on Andrew O'Shei's autonomous fetch (PR #13, `tools/fetch/`): `run.py`'s
virtual pad, the scripted pilots and the props all come from there.

## Why a new policy

The bundle's jump (`jumper.jump`) and dash (`jumper.dash`) were trained on flat ground.
Probed in the app (`run.py --stair`, `STEP_H` 0.06, 400 Hz physics), the dash hits a 6 cm
step's edge and tips past the app's tilt limit every time (3 of 3, 2026-10-22). On flat
ground it lifts the base about 0.13 m and carries it about 0.35 m.

## `jumper.step_up`

The dash, trained on a terrain of single steps (`tasks/jumper/step_up/`):

- **Same actor, same contract.** The dash's residual action, observation (167 inputs)
  and reference (`dash_flat.npz`); a board or the app can run it under the dash's name.
- **Terrain.** One box per patch, its face 0.21-0.25 m ahead of the spawn; rows by step
  height 0-0.15 m, promoted when all six feet end on the top, demoted when the rear feet
  never get near the face (`mdp/curriculum.py`).
- **Rewards.** The dash's, with the base height measured from the ground under the base
  (`ref_base_height_stepped`), plus `on_step` and `progress` (`mdp/rewards.py`).
- **Warm start** from the dash's `actor.onnx` (weights and normaliser copied into a
  checkpoint; there is no dash `.pt` to resume from).

Trained 3000 iterations, 512 envs, native CPU, ~2.2 h. Success by step height, `play=True`,
40 envs per height (model_2999: 20) (2026-10-22):

| checkpoint | 0.067 | 0.100 | 0.117 | 0.133 | 0.150 m |
|---|---|---|---|---|---|
| model_800 | 1.00 | 1.00 | 0.95 | 0.80 | 0.57 |
| model_1200 | 0.12 | 0.05 | 0.03 | 0.03 | 0.03 |
| model_2999 | 0.65 | 0.65 | 0.75 | 0.75 | 0.80 |

Training is unstable past ~800 iterations; `out/example/` is model_800.

    python scripts/train.py --task jumper.step_up --backend native --device cpu --num_envs 512 \
        --max-iterations 3000 --headless --checkpoint <warm start>.pt

## The film's take

- **Physics at 400 Hz.** `run.py --physics-hz 400`. five_foot's world steps every 5 ms;
  step_up trained at 2.5 ms, and at 5 ms the in-app hop landed about half the time. At
  400 Hz: 6 of 6 onto 10.5 cm.
- **The bundle.** A copy of `out/bundle_example` with `models/dash.onnx` replaced by
  step_up's export and `bundle.json`'s hash updated; the contracts are identical.
- **The steps** (`climb.py`, `CLIMB_N=4`): four 0.105 m steps, 0.6 m deep, the last level
  with the mattress (0.42 m). At 0.45 m deep the hop landed with the claws at the next
  step's face and pitched into it.
- **The pilot** is a list of beats (`BEATS`): walk to a fixed run-up 0.21 m short of each
  face, press L3, check every foot is over the step, retry or restart from the floor.

    CLIMB_N=4 python tools/fetch/run.py --app out/bundle_film/jumper.app --pilot tools/fetch/climb.py \
        --task jumper.five_foot --climb --scene ../maps/home.map --headless --physics-hz 400 \
        --steps 26000 --export-poses ../runs/climb_take

The first of four complete takes with no miss and no reset was kept.

## Rendering (Blender 5.2)

    Blender -b --python tools/fetch/blender/import_take.py -- --take ../runs/climb_take \
        --assets <Poly Haven assets> --blend climb.blend --furnish --mixamo <Mixamo clips> \
        --nod-at 35.8 --lie-x -1.50
    SHOTS=climb Blender -b climb.blend --python tools/fetch/blender/shots.py -- --take ../runs/climb_take --out shots

`furnish.py` dresses the steps (`climb_*` geoms) as a pouf, a stool, an ottoman and a
bench. The Mixamo character is not in this repository (Adobe's licence does not allow
redistribution); without `--mixamo` the drawn stand-in renders instead.

The training-field shot: `field_take.py` runs one robot per patch (200 here) and writes
them all as one take; `field_look.py` restyles it, `field_keep.py` drops robots a shot
cannot see (EEVEE spends ~20 s a frame syncing 16,788 objects whatever the samples).
