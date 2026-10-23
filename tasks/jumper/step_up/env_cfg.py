"""jumper.step_up's environment: jumper.dash's, on a terrain of steps.

What changes against the dash, and why:

- **The terrain.** Patches of flat ground with one box each (`mdp/terrain.py`),
  rows by height from 0 to STEP_HEIGHT_RANGE's top, columns by how far the face
  is from the spawn (FACES). The face is past the front claws at the stand
  (0.19 m ahead of the base) and short of where the dash's rear feet come down,
  so the step cannot be had by standing still or by the dash as it is.
- **Every spawn at the stand, none mid-recording.** The dash spawns at random
  phases; mid-flight states on a step terrain would start inside the box. The
  board enters the mode at the stand anyway.
- **Base height from the ground under the base** (`ref_base_height_stepped`):
  the recording's height means "above the floor the hop started on" until the
  hop is over the step, and above its top after.
- **The goal, rewarded** (`on_step`, `progress`): all six feet past the face and
  on the top. The jump's rule that nothing beyond the recording scores holds for
  a motion the recording already is; this one is the recording carried onto a
  step it never saw.
- **A step-height curriculum** (`mdp/curriculum.py`) instead of none.
"""
from __future__ import annotations

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from ..dash import env_cfg as _dash
from . import mdp as step_mdp

#: Step height at the curriculum's first and last row (m). The dash lifts the base
#: 0.13 m (probe, 2026-10-22, Mac native); 0.15 asks for a little more.
STEP_HEIGHT_RANGE = (0.0, 0.15)
#: Face distances ahead of the spawn (m), one terrain column each.
FACES = (0.21, 0.22, 0.23, 0.24, 0.25)
ROWS = 10


def env_cfg(*args, **kwargs):
    play = kwargs.get("play", False)
    cfg = _dash.env_cfg(*args, **kwargs)

    terrain = cfg.scene.terrain
    terrain.terrain_type = "generator"
    terrain.terrain_generator = TerrainGeneratorCfg(
        seed=0, curriculum=True, size=(2.0, 1.2), border_width=0.5, num_rows=ROWS, num_cols=len(FACES),
        sub_terrains={f"face_{f:.2f}": step_mdp.terrain.StepTerrainCfg(
            size=(2.0, 1.2), step_height_range=STEP_HEIGHT_RANGE, face=f) for f in FACES},
        add_lights=False,
    )
    terrain.max_init_terrain_level = 0

    # Every spawn at the stand; the recording starts on entry, after the command's delay.
    cfg.commands["jump"].rsi_fraction = 0.0
    cfg.events["reset_from_reference"].params["rsi_fraction"] = 0.0
    cfg.sim.nconmax = max(int(cfg.sim.nconmax or 0), 256)
    cfg.sim.njmax = max(int(cfg.sim.njmax or 0), 1024)

    cfg.observations["critic"].terms["step"] = ObservationTermCfg(func=step_mdp.terrain.step_obs, params={"mirror_kind": "scalar"})

    cfg.rewards["ref_base_height"] = RewardTermCfg(
        func=step_mdp.rewards.ref_base_height_stepped, weight=cfg.rewards["ref_base_height"].weight,
        params={"command_name": "jump", "sigma_sq": 0.0025},
    )
    cfg.rewards["on_step"] = RewardTermCfg(func=step_mdp.rewards.on_step, weight=4.0)
    cfg.rewards["progress"] = RewardTermCfg(func=step_mdp.rewards.progress, weight=2.0)

    cfg.curriculum = {} if play else {"step_levels": CurriculumTermCfg(func=step_mdp.curriculum.step_levels)}
    return cfg
