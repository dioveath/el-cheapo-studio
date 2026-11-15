import pytest
import trimesh
from studio.pipeline.mesh import load_geometry
from studio.schemas import GenerationSettings, VariantSettings


def test_faceted_glb_import_recovers_shared_geometry(tmp_path):
    cube = trimesh.creation.box()
    cube.unmerge_vertices()
    path = tmp_path / "cube.glb"
    cube.export(path)
    imported = load_geometry(path)
    assert len(imported.vertices) == 8
    assert imported.is_watertight


def test_point_cloud_is_not_a_surface(tmp_path):
    path = tmp_path / "points.obj"
    path.write_text("v 0 0 0\n")
    with pytest.raises(ValueError, match="empty"):
        load_geometry(path)


@pytest.mark.parametrize("settings", [GenerationSettings, VariantSettings])
def test_same_vertex_limits_for_generation_and_variants(settings):
    with pytest.raises(ValueError):
        settings(target_vertices=1)
    assert settings(target_vertices=0).target_vertices == 0
