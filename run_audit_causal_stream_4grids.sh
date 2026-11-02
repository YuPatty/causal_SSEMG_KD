#!/bin/bash
#SBATCH -A MST115015
#SBATCH -p dev
#SBATCH -J stream_4grids
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --exclude=hgpn17
#SBATCH -t 00:20:00
#SBATCH -o /work/taes10056/SSEMG-Net/logs/slurm_%j_causal_stream_4grids.out
#SBATCH -e /work/taes10056/SSEMG-Net/logs/slurm_%j_causal_stream_4grids.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'ERROR: use sbatch; no model work on login nodes' >&2; exit 1; }
PROJ=${PROJ:-/work/taes10056/SSEMG-Net}
ENV_PREFIX=${SSEMG_CONDA_PREFIX:-/home/taes10056/.conda/envs/ssemg_env_311}
PYTHON=${PYTHON:-$ENV_PREFIX/bin/python}
cd "$PROJ"
[[ -s audit_causal_stream_4grids.py ]] || { echo "ERROR: upload audit_causal_stream_4grids.py to $PROJ" >&2; exit 1; }
for F in \
    config/student_16ch_1blk_causal.yaml \
    config/student_16ch_1blk_causal_256_128.yaml \
    config/student_16ch_1blk_causal_128_32.yaml \
    config/student_16ch_1blk_causal_256_64.yaml \
    model_weight/student_16ch_1blk_respKD_causal.pth \
    model_weight/student_16ch_1blk_respKD_causal_256_128_teacher_v2_maskfix.pth \
    model_weight/student_16ch_1blk_respKD_causal_128_32_teacher_v2_maskfix_seed42.pth \
    model_weight/student_16ch_1blk_respKD_causal_256_64_teacher_v2_maskfix_dev.pth; do
    [[ -s "$F" ]] || { echo "ERROR: missing $F; wait for a completed epoch or upload the file" >&2; exit 1; }
done
module load cuda 2>/dev/null || true
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
NV_LIBS=""
if [[ -d "$ENV_PREFIX/lib/python3.11/site-packages/nvidia" ]]; then
    NV_LIBS=$(find "$ENV_PREFIX/lib/python3.11/site-packages/nvidia" -type d -name lib -print | tr '\n' ':')
fi
export LD_LIBRARY_PATH="${NV_LIBS}${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
echo "Job: $SLURM_JOB_ID Host: $(hostname); causal 512/128, 256/128, 256/64 and 128/32; inference only"
exec "$PYTHON" -u audit_causal_stream_4grids.py \
    --project-root "$PROJ" --cases 512_128 256_128 256_64 128_32 \
    --device both --seconds 20 --cpu-threads 1 --warmup-hops 5 --seed 42 \
    --output "$PROJ/results/causal_stream_4grids_j${SLURM_JOB_ID}.json"
