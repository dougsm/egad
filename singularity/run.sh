#!/bin/bash -l
#PBS -N mapelites_shapes
#PBS -l select=1:ncpus=24:mem=96GB
#PBS -l walltime=24:00:00
cd $PBS_O_WORKDIR
mkdir -p /data1/${USER}/singularity/mnt/session
singularity run --app mapelites -B /work/cyphy/doug/evolution_experiments/output:/mnt/storage -B /work/cyphy/doug:/home/root gwi.sif