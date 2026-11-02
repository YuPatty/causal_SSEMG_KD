#!/bin/bash
#SBATCH -A MST115015
#SBATCH -p normal
#SBATCH -J noKD12832_s42
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --exclude=hgpn17
#SBATCH -t 2-00:00:00
#SBATCH -o /work/taes10056/SSEMG-Net/logs/slurm_%j_noKD12832_seed42.out
#SBATCH -e /work/taes10056/SSEMG-Net/logs/slurm_%j_noKD12832_seed42.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'ERROR: use sbatch; no model work on login nodes' >&2; exit 1; }
PROJ=${PROJ:-/work/taes10056/SSEMG-Net}
DATA_ROOT=${DATA_ROOT:-/work/taes10056/semg_data_40sub/processed}
ENV_PREFIX=${SSEMG_CONDA_PREFIX:-/home/taes10056/.conda/envs/ssemg_env_311}
PYTHON=${PYTHON:-$ENV_PREFIX/bin/python}
RESUME=${RESUME:-0}
SMOKE_ONLY=${SMOKE_ONLY:-0}
[[ "$RESUME" =~ ^[01]$ && "$SMOKE_ONLY" =~ ^[01]$ ]] || { echo 'ERROR: RESUME and SMOKE_ONLY must be 0 or 1' >&2; exit 1; }
cd "$PROJ"
mkdir -p logs model_weight
STEM=student_16ch_1blk_noKD_causal_128_32_seed42
MODEL_SAVE="$PROJ/model_weight/$STEM.pth"
RESUME_PATH="$PROJ/model_weight/$STEM.resume_state.pth"
LOG_CSV="$PROJ/logs/log_$STEM.csv"
STUDENT_CFG="$PROJ/config/student_16ch_1blk_causal_128_32.yaml"
module load cuda 2>/dev/null || true
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
NV_LIBS=""
if [[ -d "$ENV_PREFIX/lib/python3.11/site-packages/nvidia" ]]; then
    NV_LIBS=$(find "$ENV_PREFIX/lib/python3.11/site-packages/nvidia" -type d -name lib -print | tr '\n' ':')
fi
export LD_LIBRARY_PATH="${NV_LIBS}${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
echo "Job: $SLURM_JOB_ID Host: $(hostname) RESUME=$RESUME SMOKE_ONLY=$SMOKE_ONLY"
echo '[experiment] causal 128/32 noKD; seed=42, batch=4, accum=4, lr=3e-4, epochs=150, patience=20'
if [[ "$SMOKE_ONLY" == 1 ]]; then
    exec "$PYTHON" -u check_student_noKD_128_32.py --project-root "$PROJ" \
        --student-config "$STUDENT_CFG" --data-root "$DATA_ROOT" --mode smoke
fi
command -v flock >/dev/null || { echo 'ERROR: flock is required' >&2; exit 1; }
exec 9>"$PROJ/model_weight/.$STEM.lock"
flock -n 9 || { echo 'ERROR: another 128/32 seed42 job is using these outputs' >&2; exit 1; }
RESUME_ARGS=()
CHECK_ARGS=()
if [[ "$RESUME" == 0 ]]; then
    for F in "$MODEL_SAVE" "$RESUME_PATH" "$LOG_CSV"; do
        [[ ! -e "$F" ]] || { echo "ERROR: output already exists: $F; use RESUME=1 to continue" >&2; exit 1; }
    done
else
    for F in "$MODEL_SAVE" "$RESUME_PATH" "$LOG_CSV"; do
        [[ -s "$F" ]] || { echo "ERROR: resume needs nonempty file: $F" >&2; exit 1; }
    done
    RESUME_ARGS=(--resume --resume_path "$RESUME_PATH")
    CHECK_ARGS=(--resume-state "$RESUME_PATH")
fi
"$PYTHON" -u check_student_noKD_128_32.py --project-root "$PROJ" \
    --student-config "$STUDENT_CFG" --data-root "$DATA_ROOT" --mode train "${CHECK_ARGS[@]}"
echo "[outputs] best=$MODEL_SAVE state=$RESUME_PATH csv=$LOG_CSV"
# Reuse the unchanged dynamic-STFT noKD branch. The fixed 256/64 check in
# this shared entry point is inside `if not no_kd`; it does not apply here.
"$PYTHON" -u pipeline_distill_crossarch_v6_256_64.py \
    --teacher_config config/config_spectrogram_v19_tt_mask.yaml \
    --student_config "$STUDENT_CFG" --data_root "$DATA_ROOT" \
    --batch_size 4 --accum 4 --num_workers 0 --epochs 150 --schedule_epochs 150 \
    --patience 20 --seed 42 --no_kd \
    --model_save "$MODEL_SAVE" --log_csv "$LOG_CSV" "${RESUME_ARGS[@]}"
