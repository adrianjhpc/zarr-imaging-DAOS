#!/bin/bash
#SBATCH -D /home/nx01/nx01/adrianj/zarr-imaging-DAOS/benchmarks
#SBATCH -o /home/nx01/nx01/adrianj/zarr-imaging-DAOS/benchmarks/lustre_zarr.%A.out
#SBATCH -e /home/nx01/nx01/adrianj/zarr-imaging-DAOS/benchmarks/lustre_zarr.%A.err
#SBATCH -J daos_zarr
#SBATCH --nodes=2
#SBATCH --nvram-option=1LM:1000

export HFI_UNIT=1

module load gnu/11.2.0
module load python3/3.12.14

export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export PYTHONPATH="/home/nx01/nx01/adrianj/zarr-DAOS/src:${PYTHONPATH}"
export LD_LIBRARY_PATH=/home/software/gnu/11.2/lib64:/home/software/anaconda/python3-2019/lib:$LD_LIBRARY_PATH

cleanup() {
    rm -fr /newlust/zarr_lustre_tmp
}
trap cleanup EXIT


mkdir -p /newlust/zarr_lustre_tmp

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
NODE_W="${NODES[0]}"
NODE_R="${NODES[1]}"
RUN_DIR="/newlust/zarr_lustre_tmp/cold_${SLURM_JOB_ID}"

echo "Writing on $NODE_W, reading cold on $NODE_R"

srun --nodes=1 --ntasks=1 -w "$NODE_W" \
    python3.12 -u benchmark_imaging_lustre.py \
        --lustre-dir "$RUN_DIR" \
        --mode write \
        --n-freq 1024

srun --nodes=1 --ntasks=1 -w "$NODE_R" \
    python3.12 -u benchmark_imaging_lustre.py \
        --lustre-dir "$RUN_DIR" \
        --mode read \
        --n-freq 1024 \
        --cleanup
