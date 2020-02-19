# EGAD

See [Project Website](https://dougsm.github.io/egad/) for overview.

## Installing

This code was developed on Ubuntu 18.04 with Python 3.7.4.

```shell script

git clone git@github.com:dougsm/egad.git
cd egad
pip3 install -e .

```

## Resizing Evaluation Objects

To resize the object meshes for a specific gripper width, use `prepare_meshes.py` in the scripts folder:

```shell script
>>> python prepare_meshes.py --help

usage: prepare_meshes.py [-h] [--stl] width path

Process meshes for specific gripper.

positional arguments:
  width       Gripper width maximum. Either meters or millimeters depending on
              desired output.
  path        Path to directory containing .obj files.

optional arguments:
  -h, --help  show this help message and exit
  --stl       Output .stl rather than .obj files (useful for printing)
```

e.g.

```shell script
python prepare_meshes.py 120 ~/Downloads/egad_eval_set --stl
```

Will resize the meshes in `~/Downloads/egad_eval_set` to the appropriate size
for a 120mm wide antipodal gripper, and output a set of .stl files to `~/Downloads/egad_eval_set/processed_meshes`

#### Note on units
Depending on the software used, it may either assume that meshes are specified in meters or millimeters.
e.g. some 3D-printing software assumes meshes are in mm, while some simulation software (e.g. VREP) assumes meters.
If your meshes appear 1000x too small or large, change the unit of the gripper width used to generate the meshes (i.e. specify 0.12 instead of 120).

