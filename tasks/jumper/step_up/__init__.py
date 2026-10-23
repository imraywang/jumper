"""jumper.step_up -- the dash aimed at a step: a forward hop that lands on top of it.

`jumper.dash`'s task on a terrain of single steps: every spawn faces a box whose
near face is a few centimetres past the front claws, and the step grows with a
terrain curriculum from flat ground to STEP_HEIGHT_RANGE's top. The policy is the
dash's shape -- the same residual, observation and contract -- so a board runs it
the way it runs the dash; what it learns is to carry the hop far enough and high
enough for all six feet to come down on the top. For the home film's climb onto
the bed, a step at a time.
"""
from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.step_up",
    assets=JUMPER_ASSETS,
    description="jumper hexapod forward hop onto a step (the dash, trained on a step-height curriculum)",
    tags=("locomotion", "jumper", "jump", "reference-tracking", "dash", "terrain"),
)
