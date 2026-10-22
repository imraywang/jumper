"""Put a Mixamo character where the take's drawn-only person was, acting what it did.

The simulation's person (tools/fetch/story.py) is a rigid stand-in with three states
-- lying on the bed, getting up (a 2 s blend), standing or walking -- and the take
records only its pose. This reads those states back from the pose and casts the
clips over them:

    lying                 Laying Sleeping, looped; Laying Nodding for the last
                          NOD_S before getting up (the crab is waving at it)
    getting up            Sitting Idle at the bed's edge, then Sit To Stand onto
                          the slippers, then Arm Stretching
    standing, still       Breathing Idle
    standing, moving      Walking (in place), played at the take's speed over the
                          clip's own, placed and turned with the take every frame

The clips are expected as <dir>/<name>.fbx, Laying Sleeping with skin and the rest
without (the files are Mixamo's and are not part of this repository). Every clip is
Mixamo's own skeleton, so one armature plays them all through its NLA.
"""
from __future__ import annotations

import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Quaternion, Vector

FPS = 30
SCALE = 0.005                    # the imports come in at twice life size; this makes Remy 1.65 m
CLIP_WALK_SPEED = 1.04           # m/s the walk cycle covers at SCALE (stride measured in its own frames)
BLEND = 8                        # frames of crossfade between clips
NOD_S = 4.0
SIT_S = 1.0
STRETCH_FRAMES = (1, 150)        # the first five seconds of the stretch: up, and arms down
#: Where the clips put the hips, in the armature's frame at SCALE, as measured on import:
#: lying on its back with the head toward +y, sitting with the feet 0.46 m ahead.
LIE_HIPS = Vector((0.0, 0.045, 1.095))
LIE_AT = Vector((-1.37, 0.72, 0.55))     # home.map's bed: hips on the mattress, head on the pillow
SIT_FEET_AHEAD = 0.46
#: The clips face -y; the take's person faces +x at yaw 0.
FACING = math.pi / 2


def _import(path: Path):
    before, acts = set(bpy.data.objects), set(bpy.data.actions)
    bpy.ops.import_scene.fbx(filepath=str(path), automatic_bone_orientation=False)
    new = set(bpy.data.objects) - before
    act = next(iter(set(bpy.data.actions) - acts))
    act.name = path.stem
    act.use_fake_user = True
    return new, act


def _states(xpos, xquat, body):
    """Per frame: 'lying' / 'rising' / 'standing', and the person's planar position and yaw."""
    n = len(xpos)
    state, pos, yaw = [], xpos[:, body, :2].astype(float), np.zeros(n)
    for i in range(n):
        w, x, y, z = (float(v) for v in xquat[i, body])
        up = 1 - 2 * (x * x + y * y)                      # the body's z axis, its world z component
        state.append("standing" if up > 0.98 else "lying" if up < 0.2 else "rising")
        yaw[i] = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    speed = np.zeros(n)
    speed[1:] = np.linalg.norm(np.diff(pos, axis=0), axis=1) * FPS
    return state, pos, yaw, speed


def _runs(flags):
    """[(start, end)] of consecutive True frames, end exclusive."""
    out, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i))
            start = None
    return out


def add_person(take_frames, bodies: list[str], clips: Path, collection) -> list[str]:
    """Add Remy; returns the body names whose drawn geoms it replaces."""
    xpos, xquat = take_frames["xpos"], take_frames["xquat"]
    body = next(i for i, name in enumerate(bodies) if name.split("/")[0] == "person")
    n = len(xpos)
    state, pos, yaw, speed = _states(xpos, xquat, body)

    new, sleep = _import(clips / "laying_sleeping.fbx")
    arm = next(o for o in new if o.type == "ARMATURE")
    arm.name = "remy"
    for o in new:
        for c in o.users_collection:
            c.objects.unlink(o)
        collection.objects.link(o)
        if o.name.startswith("Shoes"):            # barefoot: to bed, and into the slippers
            o.hide_render = o.hide_viewport = True
    acts = {"laying_sleeping": sleep}
    for name in ("laying_nodding", "sitting_idle", "sit_to_stand", "arm_stretching", "breathing_idle", "walking"):
        objs, act = _import(clips / f"{name}.fbx")
        acts[name] = act
        for o in objs:
            bpy.data.objects.remove(o, do_unlink=True)
    arm.scale = (SCALE, SCALE, SCALE)
    arm.rotation_mode = "XYZ"
    # The FBX import stands Mixamo's y-up skeleton up with a quarter turn about x; every
    # heading below is added about world z on top of it (XYZ Euler: x applies first).
    tilt = float(arm.rotation_euler.x)
    arm.animation_data_create()
    arm.animation_data.action = None

    # ── the cast: (clip, first frame, last frame exclusive, playback rate, frame range) ──
    lying = [i for i, s in enumerate(state) if s == "lying"]
    f_up = (max(lying) + 1) if lying else 0
    standing = [i for i, s in enumerate(state) if s == "standing" and i >= f_up]
    f_stand = standing[0] if standing else n
    strips = []
    f_nod = max(0, f_up - int(NOD_S * FPS))
    strips.append(("laying_sleeping", 0, f_nod, 1.0, None))
    strips.append(("laying_nodding", f_nod, f_up, 1.0, None))
    f_sit_end = f_up + int(SIT_S * FPS)
    strips.append(("sitting_idle", f_up, f_sit_end, 1.0, None))
    stand_len = int(acts["sit_to_stand"].frame_range[1] - acts["sit_to_stand"].frame_range[0])
    f_rise_end = f_sit_end + stand_len
    strips.append(("sit_to_stand", f_sit_end, f_rise_end, 1.0, None))
    f_stretch_end = f_rise_end + (STRETCH_FRAMES[1] - STRETCH_FRAMES[0])
    strips.append(("arm_stretching", f_rise_end, f_stretch_end, 1.0, STRETCH_FRAMES))
    moving = [speed[i] > 0.12 and i >= f_stretch_end for i in range(n)]
    cursor = f_stretch_end
    for a, b in _runs(moving):
        if a > cursor:
            strips.append(("breathing_idle", cursor, a, 1.0, None))
        v = float(np.median(speed[a:b]))
        strips.append(("walking", a, b, max(0.4, min(1.4, v / CLIP_WALK_SPEED)), None))
        cursor = b
    if cursor < n:
        strips.append(("breathing_idle", cursor, n, 1.0, None))

    tracks = [arm.animation_data.nla_tracks.new(), arm.animation_data.nla_tracks.new()]
    for k, (clip, a, b, rate, rng) in enumerate(strips):
        if b <= a:
            continue
        act = acts[clip]
        s0, s1 = rng if rng else (int(act.frame_range[0]), int(act.frame_range[1]))
        start = a + 1 - (BLEND if k else 0)                # Blender frames are 1-based
        strip = tracks[k % 2].strips.new(f"{k}_{clip}", max(1, start), act)
        strip.action_frame_start, strip.action_frame_end = s0, s1
        strip.scale = 1.0 / rate
        length = (s1 - s0) / rate
        want = (b + 1) - max(1, start)
        strip.repeat = max(want / length, 0.01)
        strip.blend_in = BLEND if k else 0
        strip.blend_out = BLEND
        strip.extrapolation = "NOTHING"
        strip.use_auto_blend = False

    # ── where the armature stands, frame by frame ──
    loc = np.zeros((n, 3), np.float32)
    rot = np.zeros((n, 3), np.float32)
    stand_xy = pos[f_stand] if f_stand < n else pos[-1]
    for i in range(n):
        if i < f_up:
            loc[i] = tuple(LIE_AT - LIE_HIPS)
            rot[i] = (tilt, 0.0, 0.0)
        elif i < f_rise_end:
            # Seated on the edge facing into the room, feet where the take stands up.
            loc[i] = (stand_xy[0] - SIT_FEET_AHEAD, stand_xy[1], 0.0)
            rot[i] = (tilt, 0.0, FACING)
        else:
            loc[i] = (pos[i][0], pos[i][1], 0.0)
            rot[i] = (tilt, 0.0, yaw[i] + FACING)
    for i in range(1, n):                                  # unwrap the turn, no spins at +-pi
        d = rot[i, 2] - rot[i - 1, 2]
        rot[i, 2] -= round(d / (2 * math.pi)) * 2 * math.pi
    _key_object(arm, loc, rot, n)
    print(f"[person] {len(strips)} clips; up at frame {f_up}, standing {f_stand}, "
          f"{len(_runs(moving))} walks")
    return ["person"]


def _key_object(ob, loc, rot, n):
    """Key the armature object's own transform: a second action, beside the NLA."""
    from bpy_extras import anim_utils

    ob.location = loc[0].tolist()
    ob.rotation_euler = rot[0].tolist()
    ob.keyframe_insert("location", frame=1)
    ob.keyframe_insert("rotation_euler", frame=1)
    ad = ob.animation_data
    bag = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
    frames_axis = np.arange(1, n + 1, dtype=np.float32)
    for fc in bag.fcurves:
        values = loc if fc.data_path == "location" else rot
        fc.keyframe_points.clear()
        fc.keyframe_points.add(n)
        co = np.empty(2 * n, dtype=np.float32)
        co[0::2], co[1::2] = frames_axis, values[:, fc.array_index]
        fc.keyframe_points.foreach_set("co", co)
        fc.keyframe_points.foreach_set("interpolation", [1] * n)   # LINEAR
        fc.update()
    # The object action plays over the NLA's clips, which drive only the bones.
    ad.action_blend_type = "REPLACE"
