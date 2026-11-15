from pathlib import Path


def load_geometry(path):
    import io
    import numpy as np
    import trimesh

    if path.suffix.lower() == ".obj":
        text = path.read_text(errors="strict")
        text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(("mtllib ", "usemtl ")))
        imported = trimesh.load(io.BytesIO(text.encode()), file_type="obj", force="mesh", process=False, skip_materials=True)
    else:
        imported = trimesh.load(io.BytesIO(path.read_bytes()), file_type="glb", force="mesh", process=False, skip_materials=True)
    if not isinstance(imported, trimesh.Trimesh) or not len(imported.faces) or not np.isfinite(imported.vertices).all():
        raise ValueError("The mesh is empty or has invalid coordinates.")
    if len(imported.faces) > 2_000_000:
        raise ValueError("Please import a mesh with fewer than 2 million triangles.")
    mesh = trimesh.Trimesh(vertices=np.array(imported.vertices), faces=np.array(imported.faces), process=False)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    return mesh


def topology_signature(mesh):
    import numpy as np

    edge_uses = np.bincount(mesh.edges_unique_inverse)
    return {
        "components": int(mesh.body_count),
        "euler_number": int(mesh.euler_number),
        "watertight": bool(mesh.is_watertight),
        "has_boundary": bool(np.any(edge_uses == 1)),
        "nonmanifold_edges": int(np.count_nonzero(edge_uses > 2)),
        "winding_consistent": bool(mesh.is_winding_consistent),
    }

def simplify_mesh(mesh, target_vertices):
    import numpy as np
    import pymeshlab
    import trimesh

    before = topology_signature(mesh)
    details = {"applied": False, "topology_before": before}
    if target_vertices == 0 or target_vertices >= len(mesh.vertices):
        return mesh, details

    meshset = pymeshlab.MeshSet()
    meshset.add_mesh(pymeshlab.Mesh(
        vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
        face_matrix=np.asarray(mesh.faces, dtype=np.int32),
    ))


    meshset.apply_filter("meshing_remove_duplicate_faces")
    meshset.apply_filter("meshing_remove_null_faces")
    meshset.apply_filter("meshing_remove_unreferenced_vertices")

    target_faces = max(4, round(len(mesh.faces) * target_vertices / len(mesh.vertices)))
    meshset.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=target_faces,
        qualitythr=0.7,
        preserveboundary=True,
        boundaryweight=3.0,
        preservenormal=True,
        preservetopology=True,
        planarquadric=True,
        autoclean=True,
    )
    reduced = trimesh.Trimesh(
        vertices=meshset.current_mesh().vertex_matrix(),
        faces=meshset.current_mesh().face_matrix(),
        process=False,
    )
    if not len(reduced.faces) or not np.isfinite(reduced.vertices).all():
        details["warning"] = "Reduction produced an invalid mesh; the original was kept."
        return mesh, details
    after = topology_signature(reduced)
    details["topology_after"] = after



    unsafe = (
        before["components"] != after["components"]
        or before["has_boundary"] != after["has_boundary"]
        or after["nonmanifold_edges"] > 0
        or not after["winding_consistent"]
        or (before["watertight"] and not after["watertight"])
        or (before["nonmanifold_edges"] == 0 and before["euler_number"] != after["euler_number"])
    )
    if unsafe:
        details["warning"] = "Reduction failed the connectivity or surface checks; the original was kept."
        return mesh, details
    details["applied"] = len(reduced.vertices) < len(mesh.vertices)
    details["repaired_nonmanifold_edges"] = before["nonmanifold_edges"] - after["nonmanifold_edges"]
    if len(reduced.vertices) > target_vertices * 1.2:
        details["warning"] = f"Kept {len(reduced.vertices)} vertices to retain mesh features; the target is approximate."
    return reduced, details

def export_mesh(mesh, directory, faceted=True):
    import numpy as np

    glb_path, obj_path = Path(directory) / "mesh.glb", Path(directory) / "mesh.obj"
    if not faceted:
        mesh.export(glb_path, include_normals=True)
        mesh.export(obj_path, include_texture=False)
        return str(glb_path), str(obj_path), len(mesh.vertices)

    render_mesh = mesh.copy()
    render_mesh.unmerge_vertices()
    render_mesh.vertex_normals = np.repeat(mesh.face_normals, 3, axis=0)
    render_mesh.export(glb_path, include_normals=True)
    with obj_path.open("w") as output:
        output.write("# Faceted low-poly mesh with shared geometry vertices.\ns off\n")
        for vertex in mesh.vertices:
            output.write("v " + " ".join(f"{value:.10g}" for value in vertex) + "\n")
        for normal in mesh.face_normals:
            output.write("vn " + " ".join(f"{value:.10g}" for value in normal) + "\n")
        for normal_index, face in enumerate(mesh.faces, start=1):
            output.write("f " + " ".join(f"{index + 1}//{normal_index}" for index in face) + "\n")
    return str(glb_path), str(obj_path), len(render_mesh.vertices)


def render_thumbnail(mesh, path):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    size = 512
    vertices = np.asarray(mesh.vertices, dtype=float)
    vertices = vertices - (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    forward = np.array([3.0, 2.2, 4.0])
    forward /= np.linalg.norm(forward)
    right = np.cross([0.0, 1.0, 0.0], forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    projected = vertices @ np.column_stack((right, up, forward))
    extent = np.ptp(projected[:, :2], axis=0)
    scale = size * .76 / max(float(extent.max()), 1e-12)
    points = projected[:, :2] * scale
    points -= (points.min(axis=0) + points.max(axis=0)) / 2
    points[:, 1] *= -1
    points += [size / 2, size * .47]
    light = np.array([-3.0, 5.0, 6.0])
    light /= np.linalg.norm(light)
    normals = np.asarray(mesh.face_normals).copy()
    normals[normals @ forward < 0] *= -1
    intensity = .48 + .52 * np.maximum(normals @ light, 0)
    colors = np.clip(intensity[:, None] * [218, 187, 164], 0, 255).astype(np.uint8)
    image = Image.new("RGBA", (size, size))
    shadow = ImageDraw.Draw(image)
    bottom = min(float(points[:, 1].max()) + 3, size * .9)
    shadow.ellipse((size * .25, bottom - 8, size * .75, bottom + 16), fill=(40, 24, 16, 42))
    image = image.filter(ImageFilter.GaussianBlur(12))
    draw = ImageDraw.Draw(image)
    order = np.argsort(projected[mesh.faces, 2].mean(axis=1), kind="stable")
    for face in order:
        polygon = [tuple(point) for point in points[mesh.faces[face]]]
        draw.polygon(polygon, fill=(*map(int, colors[face]), 255))
    image.resize((256, 256), Image.Resampling.LANCZOS).save(path)
