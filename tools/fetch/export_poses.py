"""Write a take out for an offline renderer: every drawn geom once, every body's pose per frame.

`run.py --export-poses DIR` records alongside (or instead of) `--video`, at the video's
frame rate, and writes

    DIR/geoms.json      one entry per drawn geom: body, type, size, local pose, rgba,
                        mesh and material; plus the model's materials and textures
    DIR/meshes.npz      vert_<id>, face_<id>, uv_<id> (where the mesh has texcoords)
    DIR/textures/<id>.png
    DIR/frames.npz      t [F], xpos [F, nbody, 3], xquat [F, nbody, 4] (w, x, y, z),
                        and the recorder's camera per frame: lookat [F, 3], distance,
                        azimuth, elevation [F] (degrees, MuJoCo's free-camera convention)

The geoms are the ones the MP4 recorder draws: those of the render model (the
appearance model when visuals are stripped), less the collision-only groups the
recorder hides. Body ids are shared between the two models, so the poses read from
the simulation's data place the render model's geoms.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _drawn_geoms(model):
    from mjrl.viewer.live import LiveViewer

    hidden = LiveViewer._collision_only_groups(model)
    for g in range(model.ngeom):
        if int(model.geom_group[g]) in hidden or model.geom_group[g] > 2:
            continue
        rgba = model.geom_rgba[g]
        mat = int(model.geom_matid[g])
        if rgba[3] <= 0.0 and (mat < 0 or model.mat_rgba[mat][3] <= 0.0):
            continue
        yield g


def dump_geoms(model, out: Path) -> None:
    import mujoco

    out.mkdir(parents=True, exist_ok=True)
    geoms, meshes, mesh_ids = [], {}, set()
    for g in _drawn_geoms(model):
        gtype = mujoco.mjtGeom(int(model.geom_type[g])).name.removeprefix("mjGEOM_").lower()
        mesh = int(model.geom_dataid[g]) if gtype == "mesh" else -1
        if mesh >= 0:
            mesh_ids.add(mesh)
        geoms.append({
            "id": g, "name": model.geom(g).name, "body": int(model.geom_bodyid[g]),
            "body_name": model.body(int(model.geom_bodyid[g])).name, "type": gtype,
            "size": [float(v) for v in model.geom_size[g]], "pos": [float(v) for v in model.geom_pos[g]],
            "quat": [float(v) for v in model.geom_quat[g]], "rgba": [float(v) for v in model.geom_rgba[g]],
            "mesh": mesh, "material": int(model.geom_matid[g]),
        })
    for m in sorted(mesh_ids):
        va, vn = int(model.mesh_vertadr[m]), int(model.mesh_vertnum[m])
        fa, fn = int(model.mesh_faceadr[m]), int(model.mesh_facenum[m])
        meshes[f"vert_{m}"] = model.mesh_vert[va:va + vn].astype(np.float32)
        meshes[f"face_{m}"] = model.mesh_face[fa:fa + fn].astype(np.int32)
        ta = int(model.mesh_texcoordadr[m])
        if ta >= 0:
            tn = int(model.mesh_texcoordnum[m])
            meshes[f"uv_{m}"] = model.mesh_texcoord[ta:ta + tn].astype(np.float32)
            meshes[f"faceuv_{m}"] = model.mesh_facetexcoord[fa:fa + fn].astype(np.int32)
    np.savez_compressed(out / "meshes.npz", **meshes)
    materials = []
    for k in range(model.nmat):
        tex = int(model.mat_texid[k][int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
        materials.append({"id": k, "name": model.material(k).name, "rgba": [float(v) for v in model.mat_rgba[k]],
                          "texture": tex, "texrepeat": [float(v) for v in model.mat_texrepeat[k]],
                          "specular": float(model.mat_specular[k]), "shininess": float(model.mat_shininess[k])})
    (out / "textures").mkdir(exist_ok=True)
    from PIL import Image

    for k in range(model.ntex):
        w, h, c = int(model.tex_width[k]), int(model.tex_height[k]), int(model.tex_nchannel[k])
        if model.tex_type[k] != int(mujoco.mjtTexture.mjTEXTURE_2D) or c not in (3, 4):
            continue
        a = int(model.tex_adr[k])
        img = model.tex_data[a:a + w * h * c].reshape(h, w, c)
        Image.fromarray(np.ascontiguousarray(img)).save(out / "textures" / f"{k}.png")
    body_names = [model.body(b).name for b in range(model.nbody)]
    (out / "geoms.json").write_text(json.dumps({"geoms": geoms, "materials": materials, "bodies": body_names},
                                               ensure_ascii=False, indent=1))


class PoseExport:
    """Sample body poses at `fps` in simulation time; `close` writes them."""

    def __init__(self, env, out: Path, fps: float, camera_of=None):
        self._env = getattr(env, "unwrapped", env)
        self._out = Path(out)
        self._fps, self._dt = float(fps), float(self._env.step_dt)
        self._camera_of = camera_of
        self._steps, self._frames = 0, 0
        self._t, self._xpos, self._xquat, self._cam = [], [], [], []
        sim = self._env.sim
        dump_geoms(getattr(sim, "render_model", sim.mj_model), self._out)
        self._grab()

    def _grab(self) -> None:
        data = self._env.sim.data
        self._t.append(self._steps * self._dt)
        self._xpos.append(data.xpos[0].detach().cpu().numpy().astype(np.float32).copy())
        self._xquat.append(data.xquat[0].detach().cpu().numpy().astype(np.float32).copy())
        cam = self._camera_of() if self._camera_of is not None else None
        self._cam.append([*cam.lookat, cam.distance, cam.azimuth, cam.elevation] if cam is not None
                         else [math.nan] * 6)
        self._frames += 1

    def update(self) -> None:
        self._steps += 1
        target = 1 + math.floor(self._steps * self._dt * self._fps + 1e-9)
        while self._frames < target:
            self._grab()

    def close(self) -> None:
        cam = np.asarray(self._cam, dtype=np.float32)
        np.savez_compressed(self._out / "frames.npz", t=np.asarray(self._t, np.float32),
                            xpos=np.stack(self._xpos), xquat=np.stack(self._xquat), fps=self._fps,
                            lookat=cam[:, :3], distance=cam[:, 3], azimuth=cam[:, 4], elevation=cam[:, 5])
        print(f"[export] {self._frames} frames of {len(self._xpos[0])} bodies to {self._out}")

    def __exit__(self, *_exc) -> bool:
        self.close()
        return False
