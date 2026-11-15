import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
import trimesh

from studio.pipeline.mesh import render_thumbnail

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path)
parser.add_argument("--rig", type=Path, required=True)
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
results, images, references = {}, [], {}
for prefix, folder in (("rig", args.rig.parent), ("animation", args.folder)):
    for extension in ("glb", "fbx"):
        path = folder / f"{prefix}.{extension}"
        bpy.ops.wm.read_factory_settings(use_empty=True)
        if extension == "glb":
            bpy.ops.import_scene.gltf(filepath=str(path))
        else:
            bpy.ops.import_scene.fbx(filepath=str(path))
        armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
        assert len(armatures) == 1
        armature = armatures[0]
        assert len(armature.data.bones) == 22
        shapes = {bone.custom_shape for bone in armature.pose.bones}
        meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj not in shapes]
        assert meshes
        for obj in meshes:
            assert any(mod.type == "ARMATURE" and mod.object == armature for mod in obj.modifiers)
            for vertex in obj.data.vertices:
                weights = [group.weight for group in vertex.groups if group.weight > 0.00001]
                assert 1 <= len(weights) <= 4 and abs(sum(weights) - 1) < 0.002
        action = armature.animation_data.action if armature.animation_data else None
        assert bool(action) == (prefix == "animation")
        frames = np.linspace(*action.frame_range, 5) if action else [1]
        poses, hips = [], []
        for frame in np.arange(frames[0], frames[-1] + 0.0001, 0.5):
            bpy.context.scene.frame_set(int(frame), subframe=float(frame % 1))
            hips.append(tuple(armature.matrix_world @ armature.pose.bones["mixamorig:Hips"].head))
        for frame in frames:
            bpy.context.scene.frame_set(int(frame), subframe=float(frame % 1))
            surfaces = []
            for obj in meshes:
                evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
                surface = evaluated.to_mesh()
                surface.calc_loop_triangles()
                vertices = np.array([evaluated.matrix_world @ vertex.co for vertex in surface.vertices])
                vertices = vertices[:, [0, 2, 1]] * [1, 1, -1]
                assert np.isfinite(vertices).all()
                surfaces.append(trimesh.Trimesh(vertices=vertices, faces=[triangle.vertices[:] for triangle in surface.loop_triangles], process=False))
                evaluated.to_mesh_clear()
            mesh = trimesh.util.concatenate(surfaces)
            poses.append(mesh.vertices)
            if extension == "glb":
                thumbnail = args.folder / f"{prefix}-{int(frame)}.png"
                render_thumbnail(mesh, thumbnail)
                image = Image.new("RGB", (256, 290), "#eee9e2")
                thumb = Image.open(thumbnail)
                image.paste(thumb, (0, 0), thumb)
                ImageDraw.Draw(image).text((12, 267), f"{prefix} · frame {frame:.1f}", fill="#392d26")
                images.append(image)
        displacement = max(float(np.max(np.linalg.norm(pose - poses[0], axis=1))) for pose in poses)
        if action:
            assert displacement > 0.01
            duration = (action.frame_range[1] - action.frame_range[0]) / (bpy.context.scene.render.fps / bpy.context.scene.render.fps_base)
            expected = json.loads((args.folder / "animation.json").read_text())["duration_seconds"]
            assert abs(duration - expected) < 0.0001
        if prefix in references:
            error = float(cKDTree(references[prefix]).query(poses[0])[0].max())
            assert error < 0.001, (path, error)
        else:
            references[prefix] = poses[0]
        results[path.name] = {"bones": 22, "vertices": len(poses[0]), "maximum_displacement": displacement,
                              "hip_vertical_range": float(np.ptp(np.asarray(hips)[:, 2])),
                              "hip_travel": float(np.linalg.norm(np.asarray(hips)[-1, :2] - np.asarray(hips)[0, :2]))}
contact = Image.new("RGB", (256 * len(images), 290), "#eee9e2")
for index, image in enumerate(images):
    contact.paste(image, (index * 256, 0))
contact.save(args.folder / "poses.png")
(args.folder / "validation.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results))
