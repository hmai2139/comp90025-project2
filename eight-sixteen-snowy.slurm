#!/bin/bash
#SBATCH --error=err-%j.err
#SBATCH --output=snowy-8-16-%j.out
#SBATCH --partition=snowy
#SBATCH --time=00:15:00
#SBATCH --nodes=8
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --job-name=project2

# You need to load a compiler before openmpi.
module load gcc/8.3.0

module load openmpi/3.1.4

echo "Scheduler 8 nodes 1 task 16 cpus (snowy)"
mpicxx -o project2 project2.cpp -fopenmp -O3
mpirun -n 8 project2 < mseq-big13-example.dat
