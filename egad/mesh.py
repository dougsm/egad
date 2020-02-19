import trimesh
import trimesh.interfaces.vhacd
import trimesh.smoothing


def voxel_to_mesh(voxels, biggest_only=True):
    vg = trimesh.voxel.VoxelGrid(voxels)
    m = vg.marching_cubes
    if biggest_only:
        m = get_biggest_mesh(m)
    return m


def get_biggest_mesh(mesh):
    try:
        return sorted(mesh.split().tolist(), key=lambda x: x.volume)[-1]
    except IndexError:
        return mesh


def scale_mesh(mesh, ratio):
    mesh.vertices *= ratio
    return mesh


def center_mesh(mesh):
    mesh.vertices -= mesh.center_mass
    return mesh


def smooth_mesh(mesh):
    trimesh.smoothing.filter_humphrey(mesh, beta=0.25)
    return mesh


def decompose_mesh(mesh, **kwargs):
    return trimesh.interfaces.vhacd.convex_decomposition(mesh, **kwargs)
