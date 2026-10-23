"""Cut a built take (import_take.py's .blend) into shots, each with its own camera, and render them.

    Blender -b ../runs/story.blend --python tools/fetch/blender/shots.py -- \\
        --take ../runs/story_take --out ../runs/story_shots --shots A,B,C

A shot is a frame range of the take, a frame step (rendering every n-th frame plays
that stretch at n times speed), and a camera: fixed (a position and a point to look
at) or following (an offset from a smoothed target, the crab, the person, or the
point between them, looking at it). Frames go to <out>/<shot>/<frame>.png, numbered
from 1 within the shot, ready for ffmpeg at 30 fps.

The targets are read from the take itself (frames.npz), not from Blender's evaluated
scene, so laying the cameras out costs no scene evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

# (name, first frame, last frame, step, camera). Frames are the take's, 1-based.
# fixed: ("fixed", position, look_at). follow: ("follow", target, offset, look_height, smoothing_s).
SHOTS = [
    ("A_night", 1, 210, 1, ("fixed", (0.55, -0.70, 1.05), (-1.10, 0.85, 0.40))),
    ("B_to_bed", 211, 370, 2, ("follow", "crab", (0.75, -0.95, 0.55), 0.15, 0.6)),
    ("C_wake", 371, 820, 1, ("fixed", (0.30, 0.25, 1.00), (-0.90, 1.08, 0.28))),
    ("D_slippers", 1030, 2965, 4, ("follow", "crab", (0.55, -0.85, 0.42), 0.10, 0.8)),
    ("E_get_up", 2966, 3300, 1, ("fixed", (1.10, -0.95, 1.40), (-0.72, 0.30, 0.92))),
    ("F_follow", 3301, 5100, 2, ("follow", "between", (1.85, -2.15, 1.55), 0.80, 1.2)),
]
#: The climb film (tools/fetch/climb.py, take climb_take_2): beats at night 0, morning 12.2,
#: hop1 18.2, the bench 32.2, onto the bed 33.9, the tap 35.8, awake 50.8, end 63.8 s.
#: "follow" with look_height None looks at the target's own height, plus the offset's z:
#: the crab climbs from the floor to the mattress inside one shot.
SHOTS_CLIMB = [
    ("A_night", 1, 300, 1, ("fixed", (1.50, -1.65, 0.55), (-0.80, -0.20, 0.30))),
    ("B_bedside", 366, 546, 1, ("follow", "crab", (0.85, -0.55, 0.45), 0.12, 0.6)),
    ("C_climb", 546, 1020, 1, ("follow", "crab", (1.05, -0.45, 0.30), None, 0.5)),
    ("D_tap", 1020, 1530, 1, ("fixed", (0.20, 0.45, 0.95), (-1.10, 1.00, 0.55))),
    ("E_morning", 1530, 1915, 1, ("fixed", (-0.55, -0.45, 1.30), (-1.15, 1.00, 0.50))),
]
#: The training field (tools/fetch/field_take.py, take field_take: 10 rows of step height
#: along x, -9.5 (flat) to 8.5 (0.15 m), 20 columns along y, -11.4 to 11.4; every robot
#: hops toward +x). A crane over the whole field, then low along the tallest row.
SHOTS_FIELD = [
    # Low over the flat end, rising along the rows toward the tall end: the steps grow as it goes.
    ("F_crane", 1, 211, 1, ("move", (-12.5, -14.0, 1.6), (-2.0, -6.0, 0.05), (-1.0, -15.5, 4.2), (6.0, -4.0, 0.05))),
    # Off the field's edge, level with the tallest row, looking down it as the robots hop.
    ("F_tall_row", 1, 211, 1, ("move", (8.15, -13.2, 0.32), (8.75, -6.5, 0.10), (8.15, -11.6, 0.30), (8.75, -4.5, 0.10))),
]
#: The open-source dash failing a 6 cm step (take flip/take_g0.22, run.py --stair at 400 Hz):
#: side on as it walks up and hops; it tips to 66 deg at frame 184, and the app resets it at 187.
SHOTS_FLIP = [
    ("G_flip", 100, 186, 1, ("fixed", (0.30, -1.95, 0.50), (0.52, -0.40, 0.16))),
]
if os.environ.get("SHOTS") == "flip":
    SHOTS = SHOTS_FLIP
elif os.environ.get("SHOTS") == "climb":
    SHOTS = SHOTS_CLIMB
elif os.environ.get("SHOTS") == "field":
    SHOTS = SHOTS_FIELD


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shots", default=None, help="comma-separated shot names or prefixes; all by default")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--stills", action="store_true", help="one frame from the middle of each shot, to check framing")
    return ap.parse_args(argv)


def targets(take: Path):
    info = json.loads((take / "geoms.json").read_text())
    f = np.load(take / "frames.npz")
    bodies = info["bodies"]
    crab = next(i for i, n in enumerate(bodies) if n.startswith("robot/"))
    person = next((i for i, n in enumerate(bodies) if n.split("/")[0] == "person"), crab)   # the field has none
    xpos = f["xpos"].astype(float)
    c, p = xpos[:, crab].copy(), xpos[:, person].copy()
    p[:, 2] = 0.9                                    # a standing person's middle, whatever the stand-in did
    return {"crab": c, "person": p, "between": 0.5 * (c + p)}


def _smooth(track: np.ndarray, seconds: float, fps: float = 30.0) -> np.ndarray:
    """A two-way exponential smoothing: no lag, no jitter from the gait."""
    if seconds <= 0:
        return track
    a = min(1.0, 1.0 / (seconds * fps))
    fwd = track.copy()
    for i in range(1, len(fwd)):
        fwd[i] = fwd[i - 1] + a * (track[i] - fwd[i - 1])
    back = fwd.copy()
    for i in range(len(back) - 2, -1, -1):
        back[i] = back[i + 1] + a * (fwd[i] - back[i + 1])
    return back


def place(cam, frame, kind, spec, tracks, smoothed, span=None):
    if kind == "fixed":
        pos, look = Vector(spec[0]), Vector(spec[1])
    elif kind == "move":
        # A camera that glides from one (position, look-at) to another across the shot, eased.
        a, b = span
        k = 0.0 if b <= a else (frame - a) / (b - a)
        k = k * k * (3 - 2 * k)
        pos = Vector(spec[0]).lerp(Vector(spec[2]), k)
        look = Vector(spec[1]).lerp(Vector(spec[3]), k)
    else:
        target, offset, height, _ = spec
        t = smoothed[target][frame - 1]
        look = Vector((t[0], t[1], height if height is not None else t[2]))
        pos = look + Vector(offset)
    cam.location = pos
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = (look - pos).to_track_quat("-Z", "Y")


def main():
    args = parse()
    sc = bpy.context.scene
    tracks = targets(args.take.resolve())
    cam = sc.camera
    cam.animation_data_clear()
    cam.data.angle_y = 0.70                           # ~40 deg vertical: a 35 mm look
    want = [s.strip() for s in args.shots.split(",")] if args.shots else None
    if sc.render.engine.startswith("BLENDER_EEVEE") and hasattr(sc.eevee, "taa_render_samples"):
        sc.eevee.taa_render_samples = args.samples
    for name, a, b, step, camspec in SHOTS:
        if want and not any(name.startswith(w) for w in want):
            continue
        kind, spec = camspec[0], camspec[1:]
        smoothed = {k: _smooth(v, spec[3]) for k, v in tracks.items()} if kind == "follow" else {}
        out = args.out.resolve() / name
        out.mkdir(parents=True, exist_ok=True)
        frames = [(a + b) // 2] if args.stills else range(a, b + 1, step)
        for k, frame in enumerate(frames, start=1):
            png = out / (f"still_{frame:05d}.png" if args.stills else f"{k:05d}.png")
            if png.exists():                          # resumable: a stopped render picks up here
                continue
            sc.frame_set(frame)
            place(cam, frame, kind, spec, tracks, smoothed, span=(a, b))
            sc.render.filepath = str(png)
            bpy.ops.render.render(write_still=True)
        print(f"[shots] {name}: frames {a}-{b} every {step} -> {out}")


main()
