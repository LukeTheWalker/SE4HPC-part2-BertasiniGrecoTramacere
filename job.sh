#!/bin/bash
#SBATCH --job-name=mpi_singularity-job
#SBATCH --output=mpi_singularity_output.txt
#SBATCH --error=mpi_singularity_error.txt # Error file
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --time=00:10:00
#SBATCH --partition=g100_all_serial

export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

export HWLOC_COMPONENTS=-gl
export OMPI_MCA_btl=^openib

# run the singularity container and map the current directory to /project
srun -N 2 singularity exec --bind /scratch_local:$TMPDIR matrix_multiplication.sif /project/main