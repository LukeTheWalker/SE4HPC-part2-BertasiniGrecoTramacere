#!/bin/bash
#SBATCH --job-name=mpi_singularity-job
#SBATCH --output=mpi_singularity_output.txt
#SBATCH --error=mpi_singularity_error.txt # Error file
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --partition=g100_all_serial

module load singularity

export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

# run the singularity container and map the current directory to /project
singularity run --bind /scratch_local:$TMPDIR matrix_multiplication.sif