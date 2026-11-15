import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pymeshlab
import trimesh

from studio.pipeline.mesh import export_mesh, simplify_mesh, topology_signature


class LowPolyTests(unittest.TestCase):
    def test_original_mode_is_unchanged(self):
        original = trimesh.creation.icosphere(subdivisions=2)
        mesh, details = simplify_mesh(original, 0)
        self.assertIs(mesh, original)
        self.assertFalse(details["applied"])

    def test_reduction_preserves_a_handle(self):
        original = trimesh.creation.torus(major_radius=2, minor_radius=0.5, major_sections=64, minor_sections=32)
        mesh, details = simplify_mesh(original, 250)
        self.assertTrue(details["applied"])
        self.assertLess(len(mesh.vertices), 350)
        self.assertEqual(topology_signature(original), topology_signature(mesh))
        self.assertEqual(mesh.euler_number, 0)
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(np.all(mesh.area_faces > 0))

    def test_reduction_preserves_separate_parts(self):
        left = trimesh.creation.icosphere(subdivisions=3)
        right = left.copy()
        right.apply_translation([4, 0, 0])
        original = trimesh.util.concatenate([left, right])
        mesh, details = simplify_mesh(original, 250)
        self.assertTrue(details["applied"])
        self.assertEqual(mesh.body_count, 2)
        self.assertEqual(topology_signature(original), topology_signature(mesh))

    def test_rejects_candidate_that_closes_a_handle(self):
        original = trimesh.creation.torus(major_radius=2, minor_radius=0.5)
        sphere = trimesh.creation.icosphere(subdivisions=1)
        candidate = pymeshlab.Mesh(vertex_matrix=sphere.vertices, face_matrix=sphere.faces)
        with patch("pymeshlab.MeshSet") as meshset:
            meshset.return_value.current_mesh.return_value = candidate
            result, details = simplify_mesh(original, 100)
        self.assertIs(result, original)
        self.assertFalse(details["applied"])
        self.assertIn("warning", details)

    def test_existing_nonmanifold_edge_repair_is_allowed(self):
        original = trimesh.creation.icosphere(subdivisions=3)
        original.faces = np.vstack([original.faces, original.faces[:1]])
        self.assertGreater(topology_signature(original)["nonmanifold_edges"], 0)
        mesh, details = simplify_mesh(original, 250)
        self.assertTrue(details["applied"])
        self.assertTrue(mesh.is_watertight)
        self.assertGreater(details["repaired_nonmanifold_edges"], 0)

    def test_flat_export_preserves_geometry_and_face_normals(self):
        original = trimesh.creation.icosphere(subdivisions=1)
        with tempfile.TemporaryDirectory() as directory:
            glb, obj, render_vertices = export_mesh(original, directory)
            self.assertEqual(render_vertices, 3 * len(original.faces))
            reloaded = trimesh.load(glb, force="mesh", process=False)
            normals = reloaded.vertex_normals[reloaded.faces]
            np.testing.assert_allclose(normals[:, 0], normals[:, 1], atol=1e-6)
            np.testing.assert_allclose(normals[:, 0], normals[:, 2], atol=1e-6)
            np.testing.assert_allclose(normals[:, 0], reloaded.face_normals, atol=1e-6)
            obj_lines = Path(obj).read_text().splitlines()
            self.assertIn("s off", obj_lines)
            self.assertEqual(sum(line.startswith("v ") for line in obj_lines), len(original.vertices))
            for filename in (glb, obj):
                loaded = trimesh.load(filename, force="mesh", process=False)
                loaded.merge_vertices(merge_norm=True, merge_tex=True)
                self.assertEqual(topology_signature(loaded), topology_signature(original))
                self.assertAlmostEqual(loaded.volume, original.volume, places=5)
            smooth_glb, _, _ = export_mesh(original, directory, faceted=False)
            smooth = trimesh.load(smooth_glb, force="mesh", process=False)
            self.assertEqual(len(smooth.vertices), len(original.vertices))


if __name__ == "__main__":
    unittest.main()


def test_thumbnail_is_geometry_based_and_transform_invariant(tmp_path):
    from PIL import Image
    from studio.pipeline.mesh import render_thumbnail

    mesh = trimesh.creation.icosphere(subdivisions=1)
    original_vertices = mesh.vertices.copy()
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    render_thumbnail(mesh, first)
    np.testing.assert_array_equal(mesh.vertices, original_vertices)
    transformed = mesh.copy()
    transformed.apply_scale(13)
    transformed.apply_translation([50, -31, 8])
    render_thumbnail(transformed, second)
    with Image.open(first) as image:
        assert image.size == (256, 256)
        assert image.mode == "RGBA"
        assert image.getpixel((0, 0))[3] == 0
        assert image.getpixel((128, 128))[3] == 255
        assert len(image.getcolors(100000)) > 16
        with Image.open(second) as other:
            assert np.abs(np.array(image).astype(float) - np.array(other)).mean() < 1
