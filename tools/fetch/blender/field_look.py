"""Restyle field_take's scene as a training ground: dark world, charcoal floor, pale steps.

    Blender -b ../runs/field.blend --python tools/fetch/blender/field_look.py

The interior HDRI the room films light with put a living room behind a 20 m field. Here
the world is a dark blue-grey, a large soft key and a rim light do the lighting, the
floor is charcoal with a faint grid (each patch is 2 x 1.2 m), and the steps are a warm
pale grey, so the red crabs and the steps' rising heights read from far away.
"""
import math

import bpy
from mathutils import Vector


def principled(name, rgb, rough):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    return m


def floor_grid(name):
    m = principled(name, (0.035, 0.038, 0.045), 0.7)
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    co = nt.nodes.new("ShaderNodeTexCoord")
    br = nt.nodes.new("ShaderNodeTexBrick")
    br.inputs["Scale"].default_value = 0.5
    br.inputs["Mortar Size"].default_value = 0.01
    br.offset = 0.0
    br.inputs["Color1"].default_value = (0.035, 0.038, 0.045, 1)
    br.inputs["Color2"].default_value = (0.04, 0.043, 0.05, 1)
    br.inputs["Mortar"].default_value = (0.10, 0.11, 0.13, 1)
    nt.links.new(co.outputs["Object"], br.inputs["Vector"])
    nt.links.new(br.outputs["Color"], b.inputs["Base Color"])
    return m


floor = floor_grid("field_floor")
step = principled("field_step", (0.62, 0.60, 0.56), 0.55)
n_floor = n_step = 0
for ob in bpy.data.objects:
    # The field's patches hang off the terrain body; a probe take (run.py --stair) has its
    # floor on the terrain and its one step on the world body.
    if ob.type != "MESH" or ob.parent is None or not ("terrain" in ob.parent.name or ob.parent.name.endswith("_world")):
        continue
    top = max((ob.matrix_world @ Vector(c)).z for c in ob.bound_box)
    ob.data.materials.clear()
    if top < 0.005:
        ob.data.materials.append(floor)
        n_floor += 1
    else:
        ob.data.materials.append(step)
        n_step += 1

for ob in list(bpy.data.objects):
    if ob.type == "LIGHT":
        bpy.data.objects.remove(ob, do_unlink=True)
world = bpy.context.scene.world
world.node_tree.nodes.clear()
bg = world.node_tree.nodes.new("ShaderNodeBackground")
out = world.node_tree.nodes.new("ShaderNodeOutputWorld")
bg.inputs["Color"].default_value = (0.012, 0.016, 0.024, 1)
bg.inputs["Strength"].default_value = 1.0
world.node_tree.links.new(bg.outputs["Background"], out.inputs["Surface"])


def light(name, kind, energy, loc, rot, size=None, color=(1, 1, 1)):
    d = bpy.data.lights.new(name, kind)
    d.energy = energy
    d.color = color
    if size is not None:
        d.size = size
    ob = bpy.data.objects.new(name, d)
    ob.location, ob.rotation_euler = loc, rot
    bpy.context.scene.collection.objects.link(ob)


light("key", "SUN", 3.2, (0, 0, 10), (math.radians(40), 0, math.radians(-60)), color=(1.0, 0.95, 0.88))
light("rim", "SUN", 1.4, (0, 0, 10), (math.radians(70), 0, math.radians(130)), color=(0.65, 0.78, 1.0))
bpy.data.lights["key"].angle = math.radians(6)
bpy.context.scene.view_settings.look = "AgX - Medium High Contrast" if "AgX - Medium High Contrast" in [
    i.identifier for i in bpy.context.scene.view_settings.bl_rna.properties["look"].enum_items] else "None"
bpy.ops.wm.save_mainfile()
print(f"[field_look] floor {n_floor}, steps {n_step}")
