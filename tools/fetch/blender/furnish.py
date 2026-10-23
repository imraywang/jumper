"""Furnish home.map's bedroom corner for the render: a modern bed, a softer sofa, decor.

home.map's furniture is boxes, and Poly Haven's beds are antiques (a rusted iron frame,
a worn day bed), so the bed is built here instead, on the map's own footprint so that
everything the simulation did beside it still lines up: an oak frame on legs, a white
mattress, an upholstered headboard, a soft pillow, and a duvet dropped by Blender's
cloth solver over the foot of the bed so it folds as cloth does. The sofa keeps the
map's shape, rounded and covered in boucle, with Poly Haven's throw pillows on it. The
rest is Poly Haven decor (CC0): a nightstand at the head of the bed on the wall side,
clear of where the crab stands to wake the sleeper, with an alarm clock and a lamp;
two potted plants; a picture over the sofa.

Everything here is drawn only: the simulation that produced the take never saw it.
"""
from __future__ import annotations

import math
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

# home.map's bed, from its collision boxes: x -1.80..-0.94, y -0.10..1.56, headboard at +y.
BED_X, BED_Y = (-1.80, -0.94), (-0.10, 1.56)
#: home.map's own mattress top (bed_mattress, z 0.27..0.42): the climb film's crab walks on it, so
#: the drawn bed has to be where the simulated one is. The alarm film's bed was 0.52, taller than
#: the map's, with nothing standing on it.
MATTRESS_TOP = 0.42
SOFA_REGION = ((-0.20, 0.90), (1.42, 1.72))


def _box(name, lo, hi, mat, bevel=0.0, segments=3, subsurf=0, collection=None):
    lo, hi = Vector(lo), Vector(hi)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(lo + hi) / 2)
    ob = bpy.context.active_object
    ob.name = name
    ob.data.transform(Matrix.Diagonal((*(hi - lo), 1.0)))
    if bevel:
        m = ob.modifiers.new("bevel", "BEVEL")
        m.width, m.segments = bevel, segments
    if subsurf:
        m = ob.modifiers.new("subsurf", "SUBSURF")
        m.levels = m.render_levels = subsurf
    for p in ob.data.polygons:
        p.use_smooth = True
    ob.data.materials.append(mat)
    if collection is not None:
        for c in ob.users_collection:
            c.objects.unlink(ob)
        collection.objects.link(ob)
    return ob


def _pillow(name, centre, size, mat, collection):
    """A box rounded into a pillow: subdivided, then pinched thin at the edges."""
    ob = _box(name, [c - s / 2 for c, s in zip(centre, size)], [c + s / 2 for c, s in zip(centre, size)], mat,
              subsurf=0, collection=collection)
    m = ob.modifiers.new("sub", "SUBSURF")
    m.levels = m.render_levels = 3
    cast = ob.modifiers.new("puff", "CAST")
    cast.factor, cast.cast_type = 0.35, "SPHERE"
    cast.use_x = cast.use_y = False                              # puff the top and bottom only
    return ob


def _duvet(mat, collection):
    """A duvet folded across the foot of the bed: a soft slab with a fold's lip and creases.

    Blender's cloth solver dropped it through the mattress in background mode, so the
    folds are drawn instead: a subdivided slab, rounded, with a fine cloud displacement
    for the creases. The sleeper's shins lie in it, which reads as tucked in.
    """
    (x0, x1), y0 = BED_X, BED_Y[0]
    slab = _box("duvet", (x0 - 0.035, y0 - 0.025, MATTRESS_TOP - 0.06), (x1 + 0.035, 0.38, MATTRESS_TOP + 0.07),
                mat, bevel=0.05, segments=6, collection=collection)
    sub = slab.modifiers.new("sub", "SUBSURF")
    sub.levels = sub.render_levels = 2
    tex = bpy.data.textures.new("duvet_creases", "CLOUDS")
    tex.noise_scale = 0.12
    disp = slab.modifiers.new("creases", "DISPLACE")
    disp.texture, disp.strength, disp.mid_level = tex, 0.018, 0.5
    return slab


def _import_model(folder: Path, collection, size=None, location=(0, 0, 0), yaw=0.0, height=None):
    """A Poly Haven glTF, scaled to `height` (m, its z extent) or uniformly by `size`, standing on z."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(next(folder.glob("*.gltf"))))
    new = [o for o in set(bpy.data.objects) - before]
    root = bpy.data.objects.new(folder.name, None)
    collection.objects.link(root)
    for o in new:
        for c in o.users_collection:
            c.objects.unlink(o)
        collection.objects.link(o)
        if o.parent is None:
            o.parent = root
    bpy.context.view_layer.update()
    pts = [o.matrix_world @ Vector(c) for o in new if o.type == "MESH" for c in o.bound_box]
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    s = (height / (hi.z - lo.z)) if height else (size or 1.0)
    centre = (lo + hi) / 2
    root.matrix_world = (Matrix.Translation(Vector(location)) @ Matrix.Rotation(yaw, 4, "Z")
                         @ Matrix.Diagonal((s, s, s, 1.0))
                         @ Matrix.Translation(Vector((-centre.x, -centre.y, -lo.z))))
    return root


def dress_steps(collection, looks, assets: Path, info: dict) -> int:
    """The climb film's steps (tools/fetch/climb.py, geoms `climb_*`) as furniture, each on its box.

    The simulated boxes are the truth -- the crab stands on their tops -- so each piece fills
    its box exactly: the pouf and the ottoman upholstered and softly rounded, the stool and the
    bench oak, the bench on legs with its top at the mattress.
    """
    tex = assets / "tex"
    mats = {
        "pouf": looks.textured("pouf", tex / "knitted_fleece", 2.0, tint=(0.86, 0.72, 0.56), colour_map=False),
        "stool": looks.textured("stool", tex / "oak_veneer_01", 1.2),
        "ottoman": looks.textured("ottoman", tex / "wool_boucle", 2.0, tint=(0.46, 0.55, 0.50), colour_map=False),
        "bench": looks.textured("bench", tex / "oak_veneer_01", 1.5),
    }
    n = 0
    for g in info["geoms"]:
        if "climb_" not in g["name"]:
            continue
        kind = g["name"].split("climb_")[-1]
        ob = bpy.data.objects.get(f"g{g['id']}")
        if ob is not None:
            bpy.data.objects.remove(ob, do_unlink=True)
        (cx, cy, cz), (hx, hy, hz) = g["pos"], g["size"]
        lo, hi = (cx - hx, cy - hy, cz - hz), (cx + hx, cy + hy, cz + hz)
        mat = mats.get(kind, mats["stool"])
        if kind in ("pouf", "ottoman"):
            _box(f"step_{kind}", lo, hi, mat, bevel=0.04 if kind == "pouf" else 0.025, segments=6,
                 collection=collection)
            if kind == "ottoman":                       # a plinth of oak under the upholstery
                _box("step_ottoman_base", (lo[0] + 0.01, lo[1] + 0.01, 0.0), (hi[0] - 0.01, hi[1] - 0.01, 0.04),
                     mats["stool"], bevel=0.004, collection=collection)
        elif kind == "bench":
            top = 0.045
            _box("step_bench_top", (lo[0], lo[1], hi[2] - top), hi, mat, bevel=0.006, collection=collection)
            for lx in (lo[0] + 0.03, hi[0] - 0.08):
                for ly in (lo[1] + 0.03, hi[1] - 0.08):
                    _box("step_bench_leg", (lx, ly, 0.0), (lx + 0.05, ly + 0.05, hi[2] - top), mat, bevel=0.004,
                         collection=collection)
            _box("step_bench_rail", (lo[0] + 0.05, lo[1] + 0.03, 0.10), (hi[0] - 0.05, lo[1] + 0.07, 0.14), mat,
                 bevel=0.004, collection=collection)
        else:                                            # the stool: a box stool with a finger slot
            _box(f"step_{kind}", lo, hi, mat, bevel=0.008, collection=collection)
        n += 1
    return n


def furnish(collection, looks, assets: Path, info: dict | None = None) -> float:
    """Replace the map's bed, dress the sofa, add decor. Returns the mattress top height."""
    tex = assets / "tex"
    boucle = looks.textured("sofa_boucle", tex / "wool_boucle", 2.5, tint=(0.60, 0.66, 0.56), colour_map=False)
    bpy.context.view_layer.update()
    # Drop the map's bed: every static mesh whose bounds sit inside its footprint.
    for ob in list(collection.objects):
        if ob.type != "MESH" or ob.parent is None or not ob.parent.name.startswith("body_0_"):
            continue
        pts = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
        c = sum(pts, Vector()) / 8
        if BED_X[0] - 0.08 <= c.x <= BED_X[1] + 0.08 and BED_Y[0] - 0.08 <= c.y <= BED_Y[1] + 0.12 and c.z < 1.2:
            bpy.data.objects.remove(ob, do_unlink=True)
        elif SOFA_REGION[0][0] <= c.x <= SOFA_REGION[1][0] and SOFA_REGION[0][1] <= c.y <= SOFA_REGION[1][1]:
            bev = ob.modifiers.new("soft", "BEVEL")
            bev.width, bev.segments, bev.limit_method = 0.035, 4, "ANGLE"
            ob.data.materials.clear()
            ob.data.materials.append(boucle)

    oak = looks.textured("oak", tex / "oak_veneer_01", 1.5)
    white = looks.textured("sheet", tex / "rough_linen", 3.0, tint=(0.93, 0.93, 0.91), colour_map=False)
    head = looks.textured("headboard", tex / "wool_boucle", 2.0, tint=(0.80, 0.75, 0.68), colour_map=False)
    fleece = looks.textured("duvet", tex / "knitted_fleece", 2.5, tint=(0.74, 0.80, 0.86), colour_map=False)
    (x0, x1), (y0, y1) = BED_X, BED_Y
    frame_top = MATTRESS_TOP - 0.20
    _box("bed_frame", (x0 - 0.02, y0 - 0.02, 0.08), (x1 + 0.02, y1 - 0.06, frame_top), oak, bevel=0.012,
         collection=collection)
    for lx in (x0 + 0.03, x1 - 0.07):
        for ly in (y0 + 0.02, y1 - 0.12):
            _box("bed_leg", (lx, ly, 0.0), (lx + 0.04, ly + 0.04, 0.08), oak, bevel=0.005, collection=collection)
    _box("mattress", (x0 + 0.01, y0 + 0.01, frame_top), (x1 - 0.01, y1 - 0.08, MATTRESS_TOP), white,
         bevel=0.045, segments=5, collection=collection)
    _box("headboard", (x0 - 0.03, y1 - 0.07, 0.22), (x1 + 0.03, y1 + 0.03, 1.02), head, bevel=0.035, segments=5,
         collection=collection)
    _pillow("pillow", ((x0 + x1) / 2, 1.30, MATTRESS_TOP + 0.06), (0.62, 0.40, 0.14), white, collection)
    _duvet(fleece, collection)
    if info is not None:
        dress_steps(collection, looks, assets, info)

    m = assets / "models"
    _import_model(m / "painted_wooden_nightstand", collection, location=(-0.62, 1.55, 0.0), yaw=math.pi, height=0.55)
    _import_model(m / "alarm_clock_01", collection, location=(-0.70, 1.52, 0.55), yaw=math.radians(200), height=0.11)
    _import_model(m / "desk_lamp_arm_01", collection, location=(-0.52, 1.60, 0.55), yaw=math.radians(160), height=0.42)
    _import_model(m / "throw_pillows_01", collection, location=(0.62, 1.28, 0.37), yaw=0.0, size=1.0)
    _import_model(m / "potted_plant_01", collection, location=(1.62, 1.45, 0.0), height=0.95)
    _import_model(m / "potted_plant_04", collection, location=(-1.70, -1.50, 0.0), height=0.65)
    _import_model(m / "hanging_picture_frame_01", collection, location=(0.62, 1.78, 1.30), yaw=0.0, height=0.55)
    return MATTRESS_TOP
