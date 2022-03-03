#!/bin/sh

#SBATCH --job-name=deepcover_batch_controller
#SBATCH --time=08:00:00
#SBATCH --partition=Teach-Standard
#SBATCH --ntasks=8

allele="HLA-A33-01"
exp_name="exp6"

# Load the default version of GNU parallel.
#module load parallel

# When running a large number of tasks simultaneously, it may be
# necessary to increase the user process limit.
ulimit -u 10000

# This specifies the options used to run srun. The "-N1 -n1" options are
# used to allocates a single core to each task.
srun="srun --exclusive -N1 -n1 --gpus=1"

# This specifies the options used to run GNU parallel:
#
#   --delay of 0.2 prevents overloading the controlling node.
#
#   -j is the number of tasks run simultaneously.
#
#   The combination of --joblog and --resume create a task log that
#   can be used to monitor progress.
#
mkdir outs
mkdir outs/$exp_name
mkdir outs/$exp_name/$allele
mkdir outs/$exp_name/$allele/logs

parallel="parallel --delay 0.2 -j $SLURM_NTASKS --joblog outs/$exp_name/$allele/logs/run_log.log --resume --resume-failed"

# Run a script, runtask.sh, using GNU parallel and srun. Parallel
# will run the runtask script for the numbers 1 through 128. To
# illustrate, the first job will run like this:
#
#   srun --exclusive -N1 -n1 ./runtask.sh arg1:1 > runtask.1

$parallel "$srun ./run_a_deepcover_batch.sh {1} > outs/$exp_name/$allele/logs/deepcover_batch.sh.{1}" ::: {0..24}

# Note that if your program does not take any input, use the -n0 option to
# call the parallel command:
#
#   $parallel -n0 "$srun ./run_noinput_task.sh > output.{1}" ::: {1..128}