"""Rebuild an exported take (tools/fetch/export_poses.py) in Blender and render it.

    /Applications/Blender.app/Contents/MacOS/Blender -b --python tools/fetch/blender/import_take.py -- \\
        --take ../runs/story_take --assets ../blender_assets --blend ../runs/story_take.blend \\
        --render ../runs/story_take_frames/ --frames 1-300 --engine EEVEE

Every drawn geom becomes an object parented to an empty per body, and every empty is
keyframed from the take's poses, so the motion is the simulation's to the frame. The
camera is the recorder's: MuJoCo's free camera (lookat, distance, azimuth, elevation)
turned into a Blender camera at the same place with the same vertical field of view.

Looks are assigned, not imported: MuJoCo's flat colours read as a toy, so geoms are
sorted by what they are -- the robot, the floor, the walls, the bed, the sofa, the
rug, the props -- and each kind gets a physically based material, textured from
Poly Haven (CC0) where a texture exists. Lighting is an interior HDRI plus a sun
through the window side of the room.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

# home.map's furniture, from its collision boxes (world metres): what a mesh of the
# imported world is, judged by where its bounding box centre falls.
REGIONS = {
    "bed": ((-1.85, -0.15, 0.02), (-0.90, 1.65, 0.80)),
    "sofa": ((-0.20, 0.90, 0.02), (1.42, 1.72, 0.70)),
    "plant": ((1.48, 1.30, 0.0), (1.80, 1.62, 1.2)),
}
FOVY = 45.0                       # MuJoCo's default vis.global.fovy, what the recorder drew with


def parse() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", type=Path, required=True)
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--blend", type=Path, default=None)
    ap.add_argument("--render", type=Path, default=None, help="directory for PNG frames")
    ap.add_argument("--frames", default=None, help="a-b (1-based, inclusive)")
    ap.add_argument("--engine", default="EEVEE", choices=("EEVEE", "CYCLES"))
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--hdri", default="cayley_interior")
    ap.add_argument("--mixamo", type=Path, default=None,
                    help="Mixamo clips (person.py): play Remy where the take's stand-in person was")
    return ap.parse_args(argv)


# ── materials ────────────────────────────────────────────────────────────────

def _principled(name, color, rough=0.5, metal=0.0, coat=0.0, sss=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color[:3], 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if "Coat Weight" in b.inputs:
        b.inputs["Coat Weight"].default_value = coat
    if sss and "Subsurface Weight" in b.inputs:
        b.inputs["Subsurface Weight"].default_value = sss
    return m


def _textured(name, folder: Path, scale=1.0, tint=None, rough_bias=0.0, colour_map=True):
    """A Principled material from Poly Haven maps, box-projected in object space."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (scale, scale, scale)
    nt.links.new(coord.outputs["Object"], mapping.inputs["Vector"])

    def image(kind, non_color):
        files = sorted(folder.glob(f"*_{kind}_2k.jpg"))
        if not files:
            return None
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = bpy.data.images.load(str(files[0]), check_existing=True)
        t.projection = "BOX"
        t.projection_blend = 0.25
        if non_color:
            t.image.colorspace_settings.name = "Non-Color"
        nt.links.new(mapping.outputs["Vector"], t.inputs["Vector"])
        return t

    diff = image("Diffuse", False) if colour_map else None
    if diff is not None:
        if tint is not None:
            mix = nt.nodes.new("ShaderNodeMix")
            mix.data_type = "RGBA"
            mix.blend_type = "MULTIPLY"
            mix.inputs["Factor"].default_value = 1.0
            mix.inputs[7].default_value = (*tint[:3], 1.0)
            nt.links.new(diff.outputs["Color"], mix.inputs[6])
            nt.links.new(mix.outputs[2], b.inputs["Base Color"])
        else:
            nt.links.new(diff.outputs["Color"], b.inputs["Base Color"])
    elif tint is not None:
        b.inputs["Base Color"].default_value = (*tint[:3], 1.0)
    rough = image("Rough", True)
    if rough is not None:
        if rough_bias:
            add = nt.nodes.new("ShaderNodeMath")
            add.operation = "ADD"
            add.inputs[1].default_value = rough_bias
            add.use_clamp = True
            nt.links.new(rough.outputs["Color"], add.inputs[0])
            nt.links.new(add.outputs[0], b.inputs["Roughness"])
        else:
            nt.links.new(rough.outputs["Color"], b.inputs["Roughness"])
    nor = image("nor_gl", True)
    if nor is not None:
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(nor.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
    return m


class Looks:
    def __init__(self, assets: Path):
        tex = assets / "tex"
        self._tex = tex
        self.cache: dict[str, bpy.types.Material] = {}
        self.floor = _textured("floor_laminate", tex / "laminate_floor_02", scale=0.6)
        self.wall = _textured("wall_plaster", tex / "painted_plaster_wall", scale=0.5, tint=(0.93, 0.90, 0.85))
        self.linen = _textured("bed_linen", tex / "rough_linen", scale=2.0, tint=(0.92, 0.92, 0.90))
        self.rug = _textured("rug_woven", tex / "rough_linen", scale=1.6, tint=_lin((0.66, 0.55, 0.40)),
                             rough_bias=0.15, colour_map=False)
        self.robot_red = _principled("crab_shell", (0.48, 0.012, 0.008), rough=0.22, coat=0.8)
        self.robot_dark = _principled("crab_dark", (0.03, 0.03, 0.035), rough=0.35, coat=0.3)
        self.robot_grey = _principled("crab_joint", (0.55, 0.56, 0.58), rough=0.4, metal=0.3)
        self.skin = _principled("skin", (0.80, 0.58, 0.47), rough=0.5, sss=0.15)

    def fabric(self, rgba):
        """Upholstery in the mesh's own colour, with the woven pattern's normal map."""
        key = "fabric_%.2f_%.2f_%.2f" % tuple(rgba[:3])
        if key not in self.cache:
            self.cache[key] = _textured(key, self._tex / "fabric_pattern_07", scale=4.0, tint=_lin(rgba))
        return self.cache[key]

    def plain(self, rgba, rough=0.5):
        key = "plain_%.2f_%.2f_%.2f_%.2f" % (*rgba[:3], rough)
        if key not in self.cache:
            self.cache[key] = _principled(key, _lin(rgba), rough=rough)
        return self.cache[key]


def _lin(rgba):
    """MuJoCo colours are display (sRGB) values; Blender's base colour is linear."""
    return [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgba[:3]]


def choose(looks: Looks, g: dict, centre: Vector, extent: Vector, mats: list[dict]) -> bpy.types.Material | None:
    """The material for a geom, or None to drop it (the map's own floor tiles: the
    textured ground plane under them took their place and the two fought)."""
    rgba = g["rgba"]
    if g["material"] >= 0 and list(np.round(rgba, 3)) == [0.5, 0.5, 0.5, 1.0]:
        rgba = mats[g["material"]]["rgba"]
    body = g["body_name"]
    if g["type"] == "plane":                  # the terrain's ground, under the whole room
        return looks.floor
    if body.startswith("robot"):
        r, gr, b = rgba[:3]
        if r > 0.5 and gr < 0.3:
            return looks.robot_red
        if max(rgba[:3]) < 0.25:
            return looks.robot_dark
        return looks.robot_grey
    if body.startswith("person"):
        return looks.skin if g["type"] == "sphere" and rgba[0] > 0.8 else looks.fabric(rgba)
    if body != "world":
        return looks.plain(rgba, rough=0.45)
    for kind, (lo, hi) in REGIONS.items():
        if all(lo[i] <= centre[i] <= hi[i] for i in range(3)):
            if kind == "bed":
                return looks.linen if max(rgba[:3]) > 0.8 else looks.fabric(rgba)
            if kind == "sofa":
                return looks.fabric(rgba)
            return looks.plain(rgba, rough=0.6)
    if g["type"] == "mesh" and extent.z < 0.03 and max(extent.x, extent.y) > 1.0:
        rug = abs(centre.x - 0.12) < 0.3 and abs(centre.y + 0.2) < 0.3 and max(extent.x, extent.y) < 2.2
        return looks.rug if rug else None
    if g["type"] == "mesh" and centre.z > 0.3:
        return looks.wall
    return looks.plain(rgba, rough=0.6)


# ── geometry ─────────────────────────────────────────────────────────────────

def _mesh_object(name, verts, faces, uvs=None, faceuv=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts.tolist(), [], faces.tolist())
    if uvs is not None and faceuv is not None:
        layer = me.uv_layers.new(name="uv")
        flat = faceuv.reshape(-1)
        uv = uvs[flat]
        uv[:, 1] = 1.0 - uv[:, 1]              # MuJoCo's texcoord row 0 is the image's top
        layer.data.foreach_set("uv", uv.reshape(-1))
    me.validate()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    if hasattr(me, "set_sharp_from_angle"):
        me.set_sharp_from_angle(angle=math.radians(35))
    return ob


def _primitive(g: dict):
    t, s = g["type"], g["size"]
    before = set(bpy.data.objects)
    if t == "box":
        bpy.ops.mesh.primitive_cube_add(size=2.0)
        ob = next(iter(set(bpy.data.objects) - before))
        ob.data.transform(Matrix.Diagonal((s[0], s[1], s[2], 1.0)))   # real size in the mesh: textures
        bev = ob.modifiers.new("bevel", "BEVEL")                           # project per metre
        bev.width = min(0.004, 0.25 * min(s[:3]))
        bev.segments = 2
    elif t == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=s[0], segments=32, ring_count=16)
        ob = next(iter(set(bpy.data.objects) - before))
    elif t == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(radius=s[0], depth=2 * s[1], vertices=32)
        ob = next(iter(set(bpy.data.objects) - before))
    elif t == "capsule":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=32, ring_count=16)
        ob = next(iter(set(bpy.data.objects) - before))
        me = ob.data
        for v in me.vertices:          # stretch a sphere's two halves apart: a capsule
            z = v.co.z
            v.co.x *= s[0]
            v.co.y *= s[0]
            v.co.z = z * s[0] + (s[1] if z > 0 else -s[1] if z < 0 else 0.0)
    elif t == "plane":
        bpy.ops.mesh.primitive_plane_add(size=2.0)
        ob = next(iter(set(bpy.data.objects) - before))
        ob.data.transform(Matrix.Diagonal((min(s[0], 6.0), min(s[1], 6.0), 1.0, 1.0)))
    else:
        return None
    for p in ob.data.polygons:
        p.use_smooth = t in ("sphere", "capsule", "cylinder")
    bpy.context.collection.objects.unlink(ob)
    return ob


def build(take: Path, looks: Looks, skip_bodies=()):
    info = json.loads((take / "geoms.json").read_text())
    meshes = np.load(take / "meshes.npz")
    frames = np.load(take / "frames.npz")
    xpos, xquat = frames["xpos"], frames["xquat"]
    coll = bpy.data.collections.new("take")
    bpy.context.scene.collection.children.link(coll)
    empties: dict[int, bpy.types.Object] = {}
    walls = []

    def empty(b):
        if b not in empties:
            e = bpy.data.objects.new(f"body_{b}_{info['bodies'][b]}", None)
            e.rotation_mode = "QUATERNION"
            coll.objects.link(e)
            empties[b] = e
        return empties[b]

    for g in info["geoms"]:
        b = g["body"]
        if g["body_name"].split("/")[0] in skip_bodies:
            continue
        if g["type"] == "mesh":
            m = g["mesh"]
            ob = _mesh_object(f"g{g['id']}", meshes[f"vert_{m}"], meshes[f"face_{m}"],
                              meshes.get(f"uv_{m}") if f"uv_{m}" in meshes else None,
                              meshes.get(f"faceuv_{m}") if f"faceuv_{m}" in meshes else None)
        else:
            ob = _primitive(g)
            if ob is None:
                continue
            ob.name = f"g{g['id']}"
        coll.objects.link(ob)
        local = Matrix.Translation(Vector(g["pos"])) @ Quaternion(g["quat"]).to_matrix().to_4x4()
        ob.matrix_basis = local @ ob.matrix_basis
        ob.parent = empty(b)
        world0 = Matrix.Translation(Vector(xpos[0, b])) @ Quaternion(xquat[0, b]).to_matrix().to_4x4()
        centre = world0 @ (ob.matrix_basis @ _bbox_centre(ob))
        extent = Vector([abs(v) for v in (ob.matrix_basis.to_3x3() @ Vector(ob.dimensions) if g["type"] == "mesh"
                                          else Vector(ob.dimensions))])
        mat = choose(looks, g, centre, extent, info["materials"])
        if mat is None:
            bpy.data.objects.remove(ob, do_unlink=True)
            continue
        ob.data.materials.clear()
        ob.data.materials.append(mat)
        if mat is looks.wall:
            walls.append(ob)

    # home.map's walls are 1.4 m: a camera at standing height sees over them into the
    # HDRI. Twice as tall, about the floor, they reach a room's 2.8 m.
    bpy.context.view_layer.update()
    for ob in walls:
        ob.parent = None
        ob.matrix_world = Matrix.Diagonal((1.0, 1.0, 2.0, 1.0)) @ ob.matrix_world
    nframes = len(xpos)
    for b, e in empties.items():
        loc, rot = xpos[:, b], xquat[:, b]
        moving = float(np.ptp(loc, axis=0).max()) > 1e-5 or float(np.ptp(rot, axis=0).max()) > 1e-5
        e.location = Vector(loc[0])
        e.rotation_quaternion = Quaternion(rot[0])
        if not moving:
            continue
        _animate(e, "location", loc, nframes)
        # Keep each quaternion on the same hemisphere as the one before, or the
        # interpolation between two keys takes the long way round.
        rot = rot.copy()
        for i in range(1, nframes):
            if np.dot(rot[i], rot[i - 1]) < 0:
                rot[i] = -rot[i]
        _animate(e, "rotation_quaternion", rot, nframes)
    return frames, nframes, info, coll


def _bbox_centre(ob) -> Vector:
    """The centre of the object's bounding box, in its own (pre-transform) space."""
    c = Vector((0, 0, 0))
    for corner in ob.bound_box:
        c += Vector(corner)
    return c / 8.0


def _animate(ob, path, values, n):
    """Key `ob.<path>` at frames 1..n from `values` [n, k], linearly.

    One keyframe_insert first, so Blender makes the action, its slot and the
    channels itself (Blender 5's layered actions bind curves through a slot, and
    curves made by hand on an action stay unbound); then every curve is refilled
    in one foreach_set.
    """
    from bpy_extras import anim_utils

    setattr(ob, path, values[0].tolist())
    ob.keyframe_insert(path, frame=1)
    ad = ob.animation_data
    bag = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
    frames_axis = np.arange(1, n + 1, dtype=np.float32)
    for fc in bag.fcurves:
        if fc.data_path != path:
            continue
        fc.keyframe_points.clear()
        fc.keyframe_points.add(n)
        co = np.empty(2 * n, dtype=np.float32)
        co[0::2], co[1::2] = frames_axis, values[:, fc.array_index]
        fc.keyframe_points.foreach_set("co", co)
        fc.keyframe_points.foreach_set("interpolation", [1] * n)   # LINEAR
        fc.update()


# ── camera, light, render ────────────────────────────────────────────────────

def camera(frames, n):
    cam_data = bpy.data.cameras.new("cam")
    cam_data.sensor_fit = "VERTICAL"
    cam_data.angle_y = math.radians(FOVY)
    cam_data.clip_start = 0.02
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.rotation_mode = "QUATERNION"
    loc = np.zeros((n, 3), np.float32)
    rot = np.zeros((n, 4), np.float32)
    for i in range(n):
        az, el = math.radians(float(frames["azimuth"][i])), math.radians(float(frames["elevation"][i]))
        fwd = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
        look = Vector(frames["lookat"][i])
        p = look - float(frames["distance"][i]) * fwd
        loc[i] = p
        q = fwd.to_track_quat("-Z", "Y")
        rot[i] = (q.w, q.x, q.y, q.z)
    for i in range(1, n):
        if np.dot(rot[i], rot[i - 1]) < 0:
            rot[i] = -rot[i]
    _animate(cam, "location", loc, n)
    _animate(cam, "rotation_quaternion", rot, n)


def lights(assets: Path, hdri: str):
    world = bpy.data.worlds.new("world")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(next((assets / "hdri" / hdri).glob("*.hdr"))))
    bg = nt.nodes["Background"]
    bg.inputs["Strength"].default_value = 0.9
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    sun_data = bpy.data.lights.new("window_sun", "SUN")
    sun_data.energy = 2.5
    sun_data.angle = math.radians(4.0)
    sun_data.color = (1.0, 0.93, 0.82)
    sun = bpy.data.objects.new("window_sun", sun_data)
    sun.rotation_euler = (math.radians(55), 0.0, math.radians(-35))
    bpy.context.scene.collection.objects.link(sun)
    fill_data = bpy.data.lights.new("ceiling", "AREA")
    fill_data.energy = 120.0
    fill_data.size = 1.5
    fill = bpy.data.objects.new("ceiling", fill_data)
    fill.location = (0.0, 0.2, 2.4)
    bpy.context.scene.collection.objects.link(fill)


def setup_render(args, n):
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = args.width, args.height
    sc.render.fps = 30
    sc.frame_start, sc.frame_end = 1, n
    if args.frames:
        a, b = (int(v) for v in args.frames.split("-"))
        sc.frame_start, sc.frame_end = a, min(b, n)
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast" if "AgX - Medium High Contrast" in \
        [i.identifier for i in sc.view_settings.bl_rna.properties["look"].enum_items] else "None"
    if args.engine == "CYCLES":
        sc.render.engine = "CYCLES"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
        sc.cycles.device = "GPU"
        sc.cycles.samples = args.samples
        sc.cycles.use_denoising = True
    else:
        sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in \
            [i.identifier for i in sc.render.bl_rna.properties["engine"].enum_items] else "BLENDER_EEVEE"
        ee = sc.eevee
        for attr, val in (("taa_render_samples", args.samples), ("use_raytracing", True),
                          ("use_shadows", True), ("use_gtao", True)):
            if hasattr(ee, attr):
                setattr(ee, attr, val)
    sc.render.image_settings.file_format = "PNG"


def main():
    args = parse()
    args.take, args.assets = args.take.resolve(), args.assets.resolve()   # images keep the path they were loaded by
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    looks = Looks(args.assets)
    frames, n, info, coll = build(args.take, looks, skip_bodies=("person",) if args.mixamo else ())
    if args.mixamo:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from person import add_person

        add_person(frames, info["bodies"], args.mixamo.resolve(), coll)
    camera(frames, n)
    lights(args.assets, args.hdri)
    setup_render(args, n)
    if args.blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(args.blend.resolve()))
    if args.render:
        bpy.context.scene.render.filepath = str(args.render.resolve()) + "/"
        bpy.ops.render.render(animation=True)


main()
