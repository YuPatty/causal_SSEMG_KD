#!/bin/bash
#SBATCH -A MST115015
#SBATCH -p dev
#SBATCH -J all_model_bench
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=hgpn17
#SBATCH -t 01:55:00
#SBATCH -o /work/taes10056/SSEMG-Net/logs/slurm_%j_all_model_bench.out
#SBATCH -e /work/taes10056/SSEMG-Net/logs/slurm_%j_all_model_bench.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "ERROR: use sbatch; no benchmarking on login nodes" >&2; exit 1; }
PROJ=${PROJ:-/work/taes10056/SSEMG-Net}
ENV_PREFIX=${SSEMG_CONDA_PREFIX:-/home/taes10056/.conda/envs/ssemg_env_311}
PYTHON=${PYTHON:-$ENV_PREFIX/bin/python}
cd "$PROJ"
for FILE in benchmark_liteemg_causal.py benchmark_teacher_v2.py benchmark_baselines.py \
    benchmark_fcn_model.py benchmark_msemg_model_source.py \
    benchmark_sdemg_sources/deep_filter_model.py benchmark_sdemg_sources/ddpm_1d.py \
    benchmark_sdemg_sources/utils.py benchmark_sdemg_sources/default.yaml; do
    [[ -s "$FILE" ]] || { echo "ERROR: missing $PROJ/$FILE; restore the previous benchmark package" >&2; exit 1; }
done
mkdir -p results
module load cuda 2>/dev/null || true
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
NV_LIBS=""
if [[ -d "$ENV_PREFIX/lib/python3.11/site-packages/nvidia" ]]; then
    NV_LIBS=$(find "$ENV_PREFIX/lib/python3.11/site-packages/nvidia" -type d -name lib -print | tr '\n' ':')
fi
export LD_LIBRARY_PATH="${NV_LIBS}${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
echo "Job: $SLURM_JOB_ID Host: $(hostname); all 9 models sequentially on one node; no training"
echo "Protocol: 10 s, FP32, batch=1, CPU threads=1, warmup=5, CPU runs=30, GPU runs=50"
exec "$PYTHON" -u benchmark_liteemg_causal.py \
    --project-root "$PROJ" --weights-dir "$PROJ/model_weight" --suite all \
    --device both --seconds 10 --cpu-threads 1 --warmup 5 --cpu-runs 30 --gpu-runs 50 \
    --output "$PROJ/results/efficiency_all_models_j${SLURM_JOB_ID}.json"
