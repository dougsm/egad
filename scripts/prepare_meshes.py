import sys
from pathlib import Path
import argparse

import trimesh
from egad.mesh import scale_mesh

parser = argparse.ArgumentParser(description='Process meshes for specific gripper.')
parser.add_argument('width', type=float,
                    help='Gripper width maximum. Either meters or millimeters depending on desired output.')
parser.add_argument('path', type=str, help='Path to directory containing .obj files.')
parser.add_argument('--stl', action='store_true', help='Output .stl rather than .obj files (useful for printing)')
args = parser.parse_args()

GRIPPER_WIDTH = args.width
GRIPPER_FRAC = 0.8
gripper_target = GRIPPER_WIDTH * GRIPPER_FRAC

ip = Path(args.path)

op = ip / 'processed_meshes'
op.mkdir(exist_ok=True)

input_meshes = ip.glob('*.obj')

for im in input_meshes:
    oname = op/im.name

    m = trimesh.load(im)
    exts = m.bounding_box_oriented.primitive.extents

    max_dim = max(list(exts))
    scale = GRIPPER_WIDTH / max_dim
    scale_mesh(m, scale)

    exts = m.bounding_box_oriented.primitive.extents

    min_dim = min(list(exts))
    if min_dim > gripper_target:
        scale = gripper_target / min_dim
        scale_mesh(m, scale)

    if args.stl:
        oname = oname.with_suffix('.stl')

    print(oname)
    m.export(oname)
