import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

MODEL_ID = "jasongzy/Make-It-Animatable"
MODEL_REVISION = "ca0daf6cb164f939e77bf32667513fc7558d5f98"
SOURCE_REVISION = "8fb51382ff6da556cdb95cc03a48200603f3a493"
TEMPLATE_REVISION = "b6f7fd42487ba75d6e9c11c09c6059daa83a67d7"
TEMPLATE_SHA256 = "002f8d269de68e5dce3d25195caf390d1aa359bbfaae3fcf4c8dc78ec36c3ba5"
CACHE = Path("/opt/mia/data/Mixamo")
TEMPLATE = CACHE / "bones.fbx"


class Rigging:
    @classmethod
    async def load(cls):
        import torch
        from huggingface_hub import snapshot_download

        if not torch.cuda.is_available():
            raise RuntimeError("Rigging requires a CUDA GPU.")
        from model import PCAE
        from util.dataset_mixamo import JOINTS_NUM, KINEMATIC_TREE

        checkpoint = Path(await asyncio.to_thread(
            snapshot_download, MODEL_ID, revision=MODEL_REVISION,
            allow_patterns=[f"output/best/new/{name}.pth" for name in ("joints_coarse", "bw", "bw_normal", "joints")],
        ))
        self = cls()
        common = dict(N=32768, input_normal=False, deterministic=True, output_dim=JOINTS_NUM)
        specifications = {
            "joints_coarse": dict(predict_bw=False, predict_joints=True, predict_joints_tail=True),
            "bw": dict(hierarchical_ratio=0.5),
            "bw_normal": dict(hierarchical_ratio=0.5, input_normal=True, input_attention=True),
            "joints": dict(hierarchical_ratio=0.5, kinematic_tree=KINEMATIC_TREE, predict_bw=False,
                           predict_joints=True, predict_joints_tail=True, joints_attn_causal=True),
        }
        self.models = {}
        for name, settings in specifications.items():
            model = PCAE(**(common | settings))
            model.load(str(checkpoint / "output/best/new" / f"{name}.pth"))
            self.models[name] = model.eval().requires_grad_(False).to("cuda")
        return self

    def predict(self, mesh, folder, faceted, progress):
        import numpy as np
        import torch
        import torch.nn.functional as F
        from pytorch3d.transforms import Transform3d
        from util.dataset_mixamo import BONES_IDX_DICT, get_hips_transform
        from util.utils import fix_random, get_normalize_transform, sample_mesh

        started = time.monotonic()
        fix_random(0)
        torch.cuda.reset_peak_memory_stats()
        mesh = mesh.copy()
        if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise ValueError("Rigging needs a valid humanoid mesh.")
        names = list(BONES_IDX_DICT)
        vertices = torch.as_tensor(np.array(mesh.vertices), dtype=torch.float32).unsqueeze(0)
        normals = torch.as_tensor(np.array(mesh.vertex_normals), dtype=torch.float32).unsqueeze(0)
        points = torch.from_numpy(sample_mesh(mesh, 32768, get_normals=True).astype(np.float32)).unsqueeze(0)
        points, point_normals = points.chunk(2, dim=-1)
        progress("Locating joints", 15)
        with torch.inference_mode():
            normalizer = get_normalize_transform(points, keep_ratio=True, recenter=True)
            points = normalizer.transform_points(points)
            vertices = normalizer.transform_points(vertices)
            coarse = self.models["joints_coarse"](points.cuda()).joints.cpu()
            rotation = Transform3d(matrix=get_hips_transform(
                coarse[:, BONES_IDX_DICT["mixamorig:Hips"], :3],
                coarse[:, BONES_IDX_DICT["mixamorig:RightUpLeg"], :3],
                coarse[:, BONES_IDX_DICT["mixamorig:LeftUpLeg"], :3],
            ).transpose(-1, -2))
            transform = normalizer.compose(rotation)
            vertices = rotation.transform_points(vertices)
            normals = F.normalize(rotation.transform_normals(normals), dim=-1)
            mesh.vertices = vertices.squeeze(0).numpy()
            tails = rotation.transform_points(coarse[..., 3:]).squeeze(0).numpy()
            points = torch.from_numpy(sample_mesh(
                mesh, 32768, get_normals=True, attn_ratio=0.5,
                attn_centers=[tails[BONES_IDX_DICT[f"mixamorig:{side}Hand"]] for side in ("Left", "Right")],
            ).astype(np.float32)).unsqueeze(0)
            points, point_normals = points.chunk(2, dim=-1)
            normalizer = get_normalize_transform(points, keep_ratio=True, recenter=False)
            points = normalizer.transform_points(points)
            vertices = normalizer.transform_points(vertices)
            transform = transform.compose(normalizer)
            mesh.vertices = vertices.squeeze(0).numpy()
            progress("Binding the surface", 40)
            weights = []
            torso = torch.tensor([i for i, name in enumerate(names) if any(part in name for part in ("Spine", "Shoulder", "Arm"))])
            for offset in range(0, vertices.shape[1], 8192):
                chunk = vertices[:, offset:offset + 8192]
                chunk_normals = normals[:, offset:offset + 8192]
                plain = self.models["bw"](points.cuda(), chunk.cuda()).bw.cpu()
                refined = self.models["bw_normal"](
                    torch.cat((points, point_normals), dim=-1).cuda(),
                    torch.cat((chunk, chunk_normals), dim=-1).cuda(),
                ).bw.cpu()
                mask = torch.isin(plain.argmax(dim=-1), torso)
                refined[mask] = plain[mask]
                weights.append(refined)
                progress("Binding the surface", 40 + 20 * min(offset + 8192, vertices.shape[1]) / vertices.shape[1])
            weights = torch.cat(weights, dim=1)
            progress("Finishing the skeleton", 65)
            joints = self.models["joints"](points.cuda()).joints.cpu()
            for side in ("Left", "Right"):
                fingers = [i for i, name in enumerate(names) if side in name and any(part in name for part in ("Thumb", "Index", "Middle", "Ring", "Pinky"))]
                weights[..., BONES_IDX_DICT[f"mixamorig:{side}Hand"]] += weights[..., fingers].sum(dim=-1)
                weights[..., fingers] = 0
            values, indices = weights.topk(4, dim=-1)
            weights.zero_().scatter_(-1, indices, values.clamp_min(0))
            sums = weights.sum(dim=-1, keepdim=True)
            if not torch.isfinite(weights).all() or not torch.isfinite(joints).all() or (sums <= 0).any():
                raise ValueError("MIA could not produce a valid humanoid rig for this mesh.")
            weights /= sums
        torch.cuda.synchronize()
        stats = {
            "model": MODEL_ID, "revision": MODEL_REVISION, "source_revision": SOURCE_REVISION, "cpu_offload": False,
            "inference_seconds": round(time.monotonic() - started, 3),
            "peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_cuda_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
            "weighted_vertices": len(mesh.vertices), "maximum_influences": 4,
        }
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / "prediction.partial"
        with temporary.open("wb") as output:
            np.savez_compressed(output, mesh=mesh, joints=joints[0, :, :3].numpy(),
                                joints_tail=joints[0, :, 3:].numpy(), bw=weights[0].numpy(),
                                bones_idx_dict=dict(BONES_IDX_DICT), transform=transform.get_matrix()[0].numpy(),
                                faceted=faceted, stats=stats)
        temporary.replace(folder / "prediction.npz")
        return stats


def template():
    import bpy

    CACHE.mkdir(parents=True, exist_ok=True)
    source = CACHE / "xbot.glb"
    if not source.is_file():
        url = f"https://raw.githubusercontent.com/mrdoob/three.js/{TEMPLATE_REVISION}/examples/models/gltf/Xbot.glb"
        with urllib.request.urlopen(url, timeout=90) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != TEMPLATE_SHA256:
            raise ValueError("The reference skeleton checksum did not match.")
        source.write_bytes(data)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = 30
    bpy.ops.import_scene.gltf(filepath=str(source))
    armature = next(obj for obj in bpy.data.objects if obj.type == "ARMATURE")
    for obj in list(bpy.data.objects):
        if obj != armature:
            bpy.data.objects.remove(obj, do_unlink=True)
    armature.animation_data_clear()
    for bone in armature.pose.bones:
        bone.matrix_basis.identity()
    bpy.context.view_layer.update()
    TEMPLATE.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.fbx(filepath=str(TEMPLATE), add_leaf_bones=False, bake_anim=False)
    TEMPLATE.with_suffix(".json").write_text(json.dumps({bone.name: bone.parent.name if bone.parent else None for bone in armature.data.bones}))
    print(json.dumps({"template": str(TEMPLATE)}))


def export(folder):
    import math
    import bpy
    import numpy as np
    from mathutils import Matrix, Vector
    from util import blender_utils

    folder = Path(folder)
    data = dict(np.load(folder / "prediction.npz", allow_pickle=True))
    blender_utils.reset()
    template_objects = blender_utils.load_file(str(TEMPLATE))
    target = next(obj for obj in template_objects if obj.type == "ARMATURE")
    target.animation_data_clear()
    target.matrix_world.identity()
    names = data["bones_idx_dict"].item()
    retained = {name: index for name, index in names.items() if not any(part in name for part in ("Thumb", "Index", "Middle", "Ring", "Pinky"))}
    indices = list(retained.values())
    names = {name: index for index, name in enumerate(retained)}
    surface = data["mesh"].item()
    geometry = bpy.data.meshes.new("Character")
    geometry.from_pydata(surface.vertices.tolist(), [], surface.faces.tolist())
    mesh = bpy.data.objects.new("Character", geometry)
    bpy.context.collection.objects.link(mesh)
    blender_utils.set_rest_bones(target, data["joints"][indices], data["joints_tail"][indices], names)
    blender_utils.set_armature_parent([mesh], target)
    groups = [mesh.vertex_groups.new(name=name) for name in names]
    weights = data["bw"][:, indices]
    for vertex, bone in zip(*np.nonzero(weights)):
        groups[bone].add([int(vertex)], float(weights[vertex, bone]), "REPLACE")
    target.matrix_world = Matrix.Rotation(math.pi / 2, 4, "X") @ Matrix(data["transform"].T).inverted()
    meshes = [mesh]
    for mesh in meshes:
        for polygon in mesh.data.polygons:
            polygon.use_smooth = not bool(data["faceted"])
    bpy.context.view_layer.update()
    for mesh in meshes:
        for vertex in mesh.data.vertices:
            if not vertex.groups or abs(sum(group.weight for group in vertex.groups) - 1) > 0.002:
                raise ValueError("The exported character has invalid skin weights.")
    bpy.ops.export_scene.gltf(filepath=str(folder / "rig.glb"), export_format="GLB", export_animations=False)
    bpy.ops.export_scene.fbx(filepath=str(folder / "rig.fbx"), add_leaf_bones=False, bake_anim=False)
    stats = data["stats"].item() | {"bones": len(target.data.bones), "bind_pose": "source"}
    temporary = folder / "rig.tmp"
    temporary.write_text(json.dumps(stats))
    temporary.replace(folder / "rig.json")
    bpy.ops.wm.read_factory_settings(use_empty=True)


def animate(rig_path, motion_path, folder):
    import bpy
    import numpy as np
    from mathutils import Matrix, Vector

    started = time.monotonic()
    folder, motion_path = Path(folder), Path(motion_path)
    settings = json.loads((folder / "settings.json").read_text())
    contacts = json.loads(motion_path.with_name("contacts.json").read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(rig_path))
    target = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
    target.animation_data_clear()
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"
              and any(modifier.type == "ARMATURE" and modifier.object == target for modifier in obj.modifiers)]
    ankle_heights = {side: (target.matrix_world @ target.data.bones[f"mixamorig:{side}Foot"].head_local).z for side in ("Left", "Right")}
    bpy.context.scene.render.fps = 30
    bpy.ops.import_anim.bvh(filepath=str(motion_path), global_scale=0.01, axis_forward="-Z", axis_up="Y",
                           use_fps_scale=False, update_scene_duration=True)
    source = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE" and obj != target)
    start, end = map(int, source.animation_data.action.frame_range)
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = start, end
    target.animation_data_create()
    action = bpy.data.actions.new(settings["prompt"][:80])
    target.animation_data.action = action
    mapping = {
        "Hips": "Hips", "Spine": "Spine1", "Spine1": "Spine2", "Spine2": "Chest", "Neck": "Neck1", "Head": "Head",
        **{side + part: side + part for side in ("Left", "Right") for part in ("Shoulder", "Arm", "ForeArm", "Hand", "Foot", "ToeBase")},
        **{side + "UpLeg": side + "Leg" for side in ("Left", "Right")},
        **{side + "Leg": side + "Shin" for side in ("Left", "Right")},
    }
    endpoints = {"Hips": "Spine1", "Chest": "Neck1", "Neck1": "Head", "Head": "HeadEnd",
                 **{side + "Hand": side + "HandMiddle1" for side in ("Left", "Right")}}
    alignment = target.matrix_world.to_quaternion().inverted() @ source.matrix_world.to_quaternion()
    up = target.matrix_world.to_quaternion().inverted() @ Vector((0, 0, 1))
    source_hips = source.data.bones["Hips"]
    target_hips = target.data.bones["mixamorig:Hips"]
    source_leg = (source_hips.head_local - source.data.bones["LeftFoot"].head_local).length
    target_leg = (target_hips.head_local - target.data.bones["mixamorig:LeftFoot"].head_local).length
    scale = target_leg / source_leg
    ordered = sorted(target.pose.bones, key=lambda bone: len(bone.parent_recursive))
    corrections = {}
    for bone in ordered:
        name = mapping[bone.name.removeprefix("mixamorig:")]
        reference = source.data.bones[name]
        endpoint = source.data.bones.get(endpoints.get(name, ""))
        if endpoint is None and reference.children:
            endpoint = min(reference.children, key=lambda child: (child.head_local - reference.tail_local).length)
        direction = ((endpoint.head_local if endpoint else reference.tail_local) - reference.head_local).normalized()
        rest_direction = (bone.bone.tail_local - bone.bone.head_local).normalized()
        corrections[bone.name] = (name, rest_direction.rotation_difference(alignment @ direction) @ bone.bone.matrix_local.to_quaternion())
    root_offsets, positions = [], {side: [] for side in ("Left", "Right")}
    initial = None
    for frame in range(start, end + 1):
        bpy.context.scene.frame_set(frame)
        for bone in ordered:
            name, correction = corrections[bone.name]
            reference = source.pose.bones[name]
            delta = reference.matrix.to_quaternion() @ reference.bone.matrix_local.to_quaternion().inverted()
            desired = alignment @ delta @ alignment.inverted() @ correction
            parent_rotation = bone.parent.matrix.to_quaternion() if bone.parent else Matrix.Identity(4).to_quaternion()
            local_rest = bone.parent.bone.matrix_local.inverted() @ bone.bone.matrix_local if bone.parent else bone.bone.matrix_local
            bone.rotation_mode = "QUATERNION"
            bone.rotation_quaternion = local_rest.to_quaternion().inverted() @ parent_rotation.inverted() @ desired
            bone.location = Vector((0, 0, 0))
            if bone.parent is None:
                displacement = alignment @ (reference.matrix.translation - source_hips.head_local) * scale
                horizontal = displacement - up * displacement.dot(up)
                if initial is None:
                    initial = horizontal.copy()
                displacement -= initial
                root_offsets.append(horizontal - initial)
                bone.location = bone.bone.matrix_local.to_3x3().inverted() @ displacement
            bone.keyframe_insert("rotation_quaternion", frame=frame, group=bone.name)
            bone.keyframe_insert("location", frame=frame, group=bone.name)
            bpy.context.view_layer.update()
        for side in positions:
            positions[side].append(tuple(target.matrix_world @ target.pose.bones[f"mixamorig:{side}Foot"].head))
    bpy.data.objects.remove(source, do_unlink=True)
    targets, constraints, masks = {}, [], {}
    for side, points in positions.items():
        points = np.asarray(points)
        indices = [index for index, name in enumerate(contacts["names"]) if name.startswith(side)]
        mask = np.max(np.asarray(contacts["contacts"])[:, indices], axis=1) > 0.5
        masks[side] = mask
        strength = np.convolve(mask.astype(float), np.ones(5) / 5, mode="same")
        locked = points.copy()
        anchor = None
        for index, planted in enumerate(mask):
            if planted and anchor is None:
                anchor = points[index].copy()
                anchor[2] = ankle_heights[side]
            if planted:
                locked[index] = anchor
            else:
                anchor = None
        empty = bpy.data.objects.new(f"{side} contact", None)
        bpy.context.collection.objects.link(empty)
        for index in range(len(points)):
            empty.location = points[index] * (1 - strength[index]) + locked[index] * strength[index]
            empty.location.z = max(empty.location.z, ankle_heights[side])
            empty.keyframe_insert("location", frame=start + index)
        shin = target.pose.bones[f"mixamorig:{side}Leg"]
        constraint = shin.constraints.new("IK")
        constraint.target = empty
        constraint.chain_count = 2
        constraint.use_stretch = False
        targets[side] = empty
        constraints.append((shin, constraint))
    baked, errors, corrected = [], [], {side: [] for side in positions}
    for frame in range(start, end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        baked.append({bone.name: bone.matrix.copy() for bone in ordered})
        for side in positions:
            point = target.matrix_world @ target.pose.bones[f"mixamorig:{side}Foot"].head
            corrected[side].append(tuple(point))
            if masks[side][frame - start]:
                errors.append((point - targets[side].location).length)
    for bone, constraint in constraints:
        bone.constraints.remove(constraint)
    for empty in targets.values():
        bpy.data.objects.remove(empty, do_unlink=True)
    for index, matrices in enumerate(baked):
        bpy.context.scene.frame_set(start + index)
        for bone in ordered:
            bone.matrix = matrices[bone.name]
            bone.keyframe_insert("rotation_quaternion", frame=start + index, group=bone.name)
            bone.keyframe_insert("location", frame=start + index, group=bone.name)
            bpy.context.view_layer.update()
        if settings["root_motion"] == "in_place":
            root = target.pose.bones["mixamorig:Hips"]
            root.location -= root.bone.matrix_local.to_3x3().inverted() @ root_offsets[index]
            root.keyframe_insert("location", frame=start + index, group=root.name)
    for unused in list(bpy.data.actions):
        if unused != action:
            bpy.data.actions.remove(unused)
    for curve in action.fcurves:
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"
    bpy.context.scene.frame_set(start)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in [target, *meshes]:
        obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(folder / "animation.glb"), export_format="GLB", export_animations=True,
                              export_animation_mode="ACTIVE_ACTIONS", export_anim_slide_to_zero=True, export_force_sampling=True,
                              export_nla_strips_merged_animation_name=action.name, use_selection=True)
    bpy.ops.export_scene.fbx(filepath=str(folder / "animation.fbx"), add_leaf_bones=False, bake_anim=True,
                            bake_anim_use_nla_strips=False, bake_anim_use_all_actions=False, use_selection=True)
    speeds = {stage: [] for stage in ("before", "after")}
    for side, mask in masks.items():
        planted = mask[1:] & mask[:-1]
        for stage, samples in (("before", positions), ("after", corrected)):
            speeds[stage].extend((np.linalg.norm(np.diff(np.asarray(samples[side])[:, :2], axis=0), axis=1) * 30)[planted].tolist())
    stats = {"frames": end - start + 1, "fps": 30, "duration_seconds": (end - start) / 30,
             "bones": len(target.data.bones), "root_motion": settings["root_motion"], "target_foot_ik": True,
             "contact_error_mean": float(np.mean(errors)) if errors else 0,
             "contact_speed_before": float(np.mean(speeds["before"])) if speeds["before"] else 0,
             "contact_speed_after": float(np.mean(speeds["after"])) if speeds["after"] else 0,
             "retarget_seconds": round(time.monotonic() - started, 2), "rig_sha256": hashlib.sha256(Path(rig_path).read_bytes()).hexdigest()}
    temporary = folder / "animation.tmp"
    temporary.write_text(json.dumps(stats))
    temporary.replace(folder / "animation.json")
    bpy.ops.wm.read_factory_settings(use_empty=True)


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1:]
    if arguments[0] == "template":
        template()
    elif arguments[0] == "export":
        export(arguments[1])

    elif arguments[0] == "animate":
        animate(*arguments[1:])
