#!/bin/bash
#SBATCH -A MST115015
#SBATCH -p normal
#SBATCH -J inferKD12832_s42
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --exclude=hgpn17
#SBATCH -t 04:00:00
#SBATCH -o /work/taes10056/SSEMG-Net/logs/slurm_%j_inferKD12832_seed42.out
#SBATCH -e /work/taes10056/SSEMG-Net/logs/slurm_%j_inferKD12832_seed42.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'ERROR: use sbatch' >&2; exit 1; }
PROJ=${PROJ:-/work/taes10056/SSEMG-Net}
DATA_ROOT=${DATA_ROOT:-/work/taes10056/semg_data_40sub/processed}
ENV_PREFIX=${SSEMG_CONDA_PREFIX:-/home/taes10056/.conda/envs/ssemg_env_311}
PYTHON=${PYTHON:-$ENV_PREFIX/bin/python}
cd "$PROJ"
mkdir -p logs results
WEIGHTS=${1:-$PROJ/model_weight/student_16ch_1blk_respKD_causal_128_32_teacher_v2_maskfix_seed42.pth}
ALIAS=${2:-student_KD_causal_128_32_teacher_v2_maskfix_seed42_best_j${SLURM_JOB_ID}}
[[ "$ALIAS" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'ERROR: invalid output alias' >&2; exit 1; }
CSV="$PROJ/results/${ALIAS}_metrics.csv"
[[ ! -e "$CSV" && ! -e "${CSV%.csv}_by_snr.csv" ]] || { echo 'ERROR: inference outputs already exist; choose another alias' >&2; exit 1; }
[[ -s "$WEIGHTS" && -s "$PROJ/test_snr_labels.json" ]] || { echo 'ERROR: missing weights or test_snr_labels.json' >&2; exit 1; }
module load cuda 2>/dev/null || true
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
NV_LIBS=""
if [[ -d "$ENV_PREFIX/lib/python3.11/site-packages/nvidia" ]]; then
    NV_LIBS=$(find "$ENV_PREFIX/lib/python3.11/site-packages/nvidia" -type d -name lib -print | tr '\n' ':')
fi
export LD_LIBRARY_PATH="${NV_LIBS}${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
"$PYTHON" -u check_student_noKD_128_32.py --project-root "$PROJ" --data-root "$DATA_ROOT" \
    --mode infer --weights "$WEIGHTS"
# This inference entry point reads model STFT dimensions from --config.
exec "$PYTHON" -u inference_student_patched_v6_256_64.py \
    --config "$PROJ/config/student_16ch_1blk_causal_128_32.yaml" \
    --weights "$WEIGHTS" --dataset "$DATA_ROOT" --batch 64 \
    --snr_labels "$PROJ/test_snr_labels.json" --csv-out "$CSV" --exp-alias "$ALIAS"
