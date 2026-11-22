#!/bin/bash
#SBATCH -D /home/nx01/nx01/adrianj/zarr-imaging-DAOS/benchmarks
#SBATCH -o /home/nx01/nx01/adrianj/zarr-imaging-DAOS/benchmarks/daos_zarr.%A.out
#SBATCH -e /home/nx01/nx01/adrianj/zarr-imaging-DAOS/benchmarks/daos_zarr.%A.err
#SBATCH -J daos_zarr
#SBATCH --nodes=2
#SBATCH --nvram-option=1LM:1000

set -e

export HFI_UNIT=1

module load gnu/11.2.0
module load python3/3.12.14

export DAOS_AGENT_DRPC_DIR=/var/run/daos_agent
export DAOS_POOL=tutorial
export DAOS_CONT="zarr_benchmark_${SLURM_JOB_ID}"
export DAOS_POSIX_CONT="zarr_array_${SLURM_JOB_ID}"
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export PYTHONPATH="/home/nx01/nx01/adrianj/zarr-DAOS/src:${PYTHONPATH}"

cleanup() {
    daos cont destroy "$DAOS_POOL" "$DAOS_CONT" --force || true
    daos cont destroy "$DAOS_POOL" "$DAOS_POSIX_CONT" --force || true
}
trap cleanup EXIT

daos cont create "$DAOS_POOL" "$DAOS_CONT" --type=PYTHON
daos cont create "$DAOS_POOL" "$DAOS_POSIX_CONT" --type=POSIX

export LD_LIBRARY_PATH=/home/software/gnu/11.2/lib64:/home/software/anaconda/python3-2019/lib:$LD_LIBRARY_PATH

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
NODE_W="${NODES[0]}"
NODE_R="${NODES[1]}"
RUN_ID="cold_${SLURM_JOB_ID}"

echo "Writing on $NODE_W, reading cold on $NODE_R"

ZARR_IMAGING_PROFILE=1 srun --nodes=1 --ntasks=1 -w "$NODE_W" \
    python3.12 -u benchmark_imaging.py \
        --pool tutorial \
        --cont "zarr_benchmark_${SLURM_JOB_ID}" \
        --array-cont "zarr_array_${SLURM_JOB_ID}" \
        --backend both \
        --mode write \
        --run-id "$RUN_ID" \
        --n-freq 1024

ZARR_IMAGING_PROFILE=1 srun --nodes=1 --ntasks=1 -w "$NODE_R" \
    python3.12 -u benchmark_imaging.py \
        --pool tutorial \
        --cont "zarr_benchmark_${SLURM_JOB_ID}" \
        --array-cont "zarr_array_${SLURM_JOB_ID}" \
        --backend both \
        --mode read \
        --run-id "$RUN_ID" \
        --n-freq 1024
