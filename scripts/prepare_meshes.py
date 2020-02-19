import sys
import subprocess
from pathlib import Path

import trimesh

from gwi.cppn.mesh import scale_mesh


GRIPPER_WIDTH = 100  # in mm
GRIPPER_FRAC = 1.0
gripper_target = GRIPPER_WIDTH * GRIPPER_FRAC


ip = Path(sys.argv[1])
op = ip / 'processed_meshes'
op.mkdir(exist_ok=True)


script = Path(__file__).resolve().parent / 'resample.mlx'


input_meshes = ip.glob('*.obj')


for im in input_meshes:
    oname = op/im.name
    subprocess.run([
        'meshlabserver',
        '-i', im,
        '-o', oname,
        '-s', script
    ])

    m = trimesh.load(oname)
    exts = m.bounding_box_oriented.primitive.extents

    max_dim = max(list(exts))
    scale = GRIPPER_WIDTH / max_dim
    scale_mesh(m, scale)

    exts = m.bounding_box_oriented.primitive.extents

    min_dim = min(list(exts))
    if min_dim > gripper_target:
        scale = gripper_target / min_dim
        scale_mesh(m, scale)

    m.export(oname)
    # m.export(oname.with_suffix('.stl'))

