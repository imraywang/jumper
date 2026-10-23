"""Save a copy of the field scene with only the robots a shot can see.

    Blender -b ../runs/field.blend --python tools/fetch/blender/field_keep.py -- xmin xmax ymin ymax out.blend

16,788 objects cost EEVEE ~20 s a frame to sync whatever the samples (2026-10-23, M4 Pro);
a robot outside the box (its base's start position) is deleted, its geoms and empties both.
"""
import sys

import bpy

xmin, xmax, ymin, ymax = (float(v) for v in sys.argv[sys.argv.index("--") + 1:][:4])
out = sys.argv[sys.argv.index("--") + 5]
bpy.context.scene.frame_set(1)
drop = []
for e in bpy.data.objects:
    if e.type == "EMPTY" and e.name.startswith("body_") and e.name.endswith("base_link") and "robot" in e.name:
        x, y = e.matrix_world.translation.x, e.matrix_world.translation.y
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            drop.append(e.name.split("_", 2)[2].rsplit("/", 1)[0])      # e.g. "robot#12"
prefixes = tuple(p + "/" for p in drop)
doomed = set()
for e in bpy.data.objects:
    if e.type == "EMPTY" and e.name.startswith("body_") and e.name.split("_", 2)[2].startswith(prefixes):
        doomed.add(e.name)
        doomed.update(c.name for c in e.children_recursive)
for name in doomed:                    # by name: a child may also be a body's empty
    ob = bpy.data.objects.get(name)
    if ob is not None:
        bpy.data.objects.remove(ob, do_unlink=True)
gone = len(doomed)
bpy.ops.wm.save_as_mainfile(filepath=out)
print(f"[field_keep] dropped {len(drop)} robots ({gone} objects), {len(bpy.data.objects)} left -> {out}")
