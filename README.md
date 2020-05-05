# EGAD!

EGAD! an Evolved Grasping Analysis Dataset for diversity and reproducibility in robotic manipulation

Accepted to IEEE RA-L, April 2020.  [[arXiv preprint](https://arxiv.org/abs/2003.01314)]

## Dataset Download

See the [Project Website](https://dougsm.github.io/egad/) for overview, dataset download and multimedia material.


## Installing as a Library

This code was developed on Ubuntu 18.04 with Python 3.7.4.

```shell script
git clone https://github.com/dougsm/egad.git 
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

---

## Generating an Object Dataset

We provide a Singularity container (similar to Docker but optimised for HPC clusters) to ease reproduction and deployment of our code.
This helps streamline the deployment of dependencies and also allows for easy deployment on computing clusters.

Note that the code will take a *very* long time to run on a desktop PC.  
The evaluation step can be largely parallelised, so running on a computer with many cores is recommended.


**1. Install Singularity**

Follow the installation instructions at [https://sylabs.io/docs/](https://sylabs.io/docs/).
This code has been run and tested with Singularity 3.5.

**2. Clone this repo**

```shell script
git clone https://github.com/dougsm/egad.git
cd egad
pip3 install -e .
```

**3. Build the singularity container**

```shell script
cd singularity
sudo singularity build egad.sif singularity.def 

# You can test that the compile worked by running a shell inside the container.
singularity run egad.sif

# exit with ctrl+D
```

Once built, the container can be deployed on any other machine running Singularity, e.g. a HPC cluster.

**4. Run with the default arguments**

Running the `datasetgen` app of the container will run the object generation script.

The singularity container requires a directory on the host machine to output data,
which is mapped internally to `/output`.
Specify this with `-B /path/to/local/output/directory:/output`, replacing this with
a path on your machine.

```shell script
singularity run -B /path/to/local/output/directory:/output --app datasetgen egad.sif
```

If running correctly, updates will be printed to the screen.

Data will appear in a timestamped subdirectory of the output directory with the following structure:

```
<timestamp>
│   gen_<gen>.json 
│   (JSON dict of current state of the search space as of generation <gen>. 
│    Keys = position in map "(<complexity>, <difficulty>)".
│    Values = list of mesh ids, corresponding to files in the pool directory)
│
└───pool
│       <generation>_<uniqueid>.ext 
│       (raw outputs e.g. meshes, reeb graphs, thumbnails)
│
└───checkpoints
│       <checkpoints for the NEAT algorithm>
│
└───diversity
│       gen_div_<gen>.json
│       (current diversity of every mesh as of generation <gen>)
```

**5. Visualise results**

(On the host machine)

This script will populate the `/path/to/local/output/directory/<timestamp>/viz` directory
with a visualisation of the algorithm's progress at each generation.

```shell script
cd <path/to/egad>/scripts
python draw_map.py /path/to/local/output/directory/<timestamp>
```

**6. Running with custom arguments**

Arguments can be provided to the script by appending them to the above singularity command.
For example, one can change the size of the search space, or the number of parallel processes used for evaluation.  

A full list of arguments can be viewed by providing the `--help` tag:

```shell script
singularity run -B --app datasetgen egad.sif --help

usage: generate_dataset.py [-h] [--cppnsize CPPNSIZE]
                           [--cppnconfig CPPNCONFIG] [--mapsize MAPSIZE]
                           [--complexitydims COMPLEXITYDIMS]
                           [--difficultydims DIFFICULTYDIMS]
                           [--cellsize CELLSIZE] [--neighbours NEIGHBOURS]
                           [--divthreshold DIVTHRESHOLD]
                           [--processes PROCESSES] [--chunksize CHUNKSIZE]
                           [--reebpath REEBPATH] [--dexnetenv DEXNETENV]
                           [--outputdir OUTPUTDIR] [--resume]

Generate shape dataset using CPPNs.

optional arguments:
  -h, --help            show this help message and exit
  --cppnsize CPPNSIZE   Dimension of CPPN voxel grid.
  --cppnconfig CPPNCONFIG
                        path to CPPN config file
  --mapsize MAPSIZE     Dimension of the search space
  --complexitydims COMPLEXITYDIMS
                        Number of cells in complexity dimension
  --difficultydims DIFFICULTYDIMS
                        Number of cells in difficulty dimension
  --cellsize CELLSIZE   Meshes to keep per cell
  --neighbours NEIGHBOURS
                        Number of neighbours to compute diversity
  --divthreshold DIVTHRESHOLD
                        Minimum threshold for object diversity
  --processes PROCESSES
                        Number of procesing threads
  --chunksize CHUNKSIZE
                        Number of jobs per thread at a time
  --reebpath REEBPATH   Path to reeb graph implementation
  --dexnetenv DEXNETENV
                        Python environment for dexnet
  --outputdir OUTPUTDIR
  --resume              resume from last run

```

**7. Running with local/custom code**

NB: If your custom code requires extra dependencies these will have to be build into the singularity container
by modifiying `singularity/singularity.def`, as the container filesystem can not be modified after compile.

Inside the container, the egad code is pulled from github and stored in `/home/co/egad`.
You can overwrite this with code the host machine using the `-B` bind command to the container.
This way, you can test new code without having to rebuild the container.

```shell script
# e.g. to overwrite the entire directory with a local copy.
singularity run -B /path/to/local/output/directory:/output -B /home/user/dev/egad/:/home/co/egad --app datasetgen egad.sif

# e.g. to overwrite a single file (scripts/generate_dataset.py in this case)
singularity run -B /path/to/local/output/directory:/output -B /home/user/dev/egad/scripts/generate_dataset.py:/home/co/egad/scripts/generate_dataset.py --app datasetgen egad.sif
```

---

## References and Acknowledgements

This project wouldn't have been possible without the following projects:

- Dex-Net: [https://github.com/BerkeleyAutomation/dex-net](https://github.com/BerkeleyAutomation/dex-net)
- Reeb Graphs: [https://github.com/dbespalov/reeb_graph](https://github.com/dbespalov/reeb_graph)
- NEAT-Python: [https://neat-python.readthedocs.io/en/latest/](https://neat-python.readthedocs.io/en/latest/)
- PyTorch-NEAT: [https://github.com/dougsm/PyTorch-NEAT](https://github.com/dougsm/PyTorch-NEAT), originally [https://github.com/uber-research/PyTorch-NEAT](https://github.com/uber-research/PyTorch-NEAT)
- Trimesh: [https://trimsh.org/index.html](https://trimsh.org/index.html)
- V-HACD: [https://github.com/kmammou/v-hacd](https://github.com/kmammou/v-hacd)