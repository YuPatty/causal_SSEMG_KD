#!/bin/bash
#SBATCH -A MST115015
#SBATCH -p normal
#SBATCH -J kd25664_timeKD
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --exclude=hgpn17
#SBATCH -t 2-00:00:00
#SBATCH -o /work/taes10056/SSEMG-Net/logs/slurm_%j_kd25664_timeKD.out
#SBATCH -e /work/taes10056/SSEMG-Net/logs/slurm_%j_kd25664_timeKD.err

# New experiment only. W_TIME=0.25 is an untuned starting trial, not an optimum.
# Fresh:  sbatch --export=ALL,RESUME=0,W_TIME=0.25,SEED=42 <this script>
# Resume: sbatch --export=ALL,RESUME=1,W_TIME=0.25,SEED=42 <this script>
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'ERROR: use sbatch; do not train or run Torch preflight on a login node.' >&2
    exit 1
fi

PROJ=${PROJ:-/work/taes10056/SSEMG-Net}
DATA_ROOT=${DATA_ROOT:-/work/taes10056/semg_data_40sub/processed}
# Do not accidentally inherit an unrelated login/base Conda environment.
export CONDA_PREFIX=${SSEMG_CONDA_PREFIX:-/home/taes10056/.conda/envs/ssemg_env_311}
PYTHON=${PYTHON:-$CONDA_PREFIX/bin/python}
RESUME=${RESUME:-0}
W_TIME=${W_TIME:-0.25}
SEED=${SEED:-42}
SMOKE_ONLY=${SMOKE_ONLY:-0}
if [[ ! "$W_TIME" =~ ^[0-9]+([.][0-9]+)?$ || ! "$SEED" =~ ^[0-9]+$ ]]; then
    echo 'ERROR: W_TIME must be a nonnegative decimal (e.g. 0.25); SEED must be an integer.' >&2
    exit 1
fi
if [[ "$SMOKE_ONLY" != 0 && "$SMOKE_ONLY" != 1 ]]; then
    echo 'ERROR: SMOKE_ONLY must be 0 or 1' >&2
    exit 1
fi
cd "$PROJ"
mkdir -p logs model_weight
STEM=student_16ch_1blk_respKD_causal_256_64_teacher_v2_timeKD_w${W_TIME//./p}_seed${SEED}
MODEL_SAVE="$PROJ/model_weight/$STEM.pth"
RESUME_PATH="$PROJ/model_weight/$STEM.resume_state.pth"
LOG_CSV="$PROJ/logs/log_$STEM.csv"
TEACHER_CFG="$PROJ/config/config_spectrogram_v19_tt_mask.yaml"
TEACHER_WEIGHTS="$PROJ/model_weight/v19_tt_mask_v2_mix_nv0_weights.pth"
STUDENT_CFG="$PROJ/config/student_16ch_1blk_causal_256_64.yaml"

RESUME_ARGS=()
if [[ "$SMOKE_ONLY" == 0 ]]; then
    command -v flock >/dev/null || { echo 'ERROR: flock is required' >&2; exit 1; }
    exec 9>"$PROJ/model_weight/.$STEM.lock"
    flock -n 9 || { echo 'ERROR: another job is writing this same experiment; refusing concurrent run' >&2; exit 1; }
    case "$RESUME" in
        0)
            if [[ -e "$MODEL_SAVE" || -e "$RESUME_PATH" || -e "$LOG_CSV" ]]; then
                echo 'ERROR: fresh run refuses to overwrite existing outputs. Use RESUME=1 only for this same run.' >&2
                exit 1
            fi
            ;;
        1)
            if [[ ! -s "$MODEL_SAVE" || ! -s "$RESUME_PATH" || ! -s "$LOG_CSV" ]]; then
                echo 'ERROR: resume requires this time-KD experiment own nonempty best weights, state and CSV.' >&2
                exit 1
            fi
            RESUME_ARGS=(--resume --resume_path "$RESUME_PATH")
            ;;
        *) echo 'ERROR: RESUME must be 0 or 1' >&2; exit 1 ;;
    esac
fi

# Compute-node preflight; all Python/Torch work is after the Slurm-job guard.
[[ -x "$PYTHON" ]] || { echo "ERROR: missing Python: $PYTHON" >&2; exit 1; }
module load cuda 2>/dev/null || true
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
NV_LIBS=""
if [[ -d "$CONDA_PREFIX/lib/python3.11/site-packages/nvidia" ]]; then
    NV_LIBS=$(find "$CONDA_PREFIX/lib/python3.11/site-packages/nvidia" -type d -name lib -print | tr '\n' ':')
fi
export LD_LIBRARY_PATH="${NV_LIBS}${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
echo "Job: $SLURM_JOB_ID  Host: $(hostname)  RESUME=$RESUME  SMOKE_ONLY=$SMOKE_ONLY"
echo "[experiment] Teacher 512/128 -> causal Student 256/64; seed=$SEED batch=4 accum=4 time_weight=$W_TIME"
echo "Best weights: $MODEL_SAVE"
echo "Resume state: $RESUME_PATH"
echo "CSV: $LOG_CSV"

"$PYTHON" - "$PROJ" "$DATA_ROOT" "$SMOKE_ONLY" <<'PY'
import hashlib
import os
import sys
from pathlib import Path
import torch
import yaml

root, data, smoke = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
required = (
    'pipeline_distill_crossarch_v6_256_64_timeKD.py',
    'distill_loss_v3_256_64_timeKD.py', 'distill_loss_v3_256_64.py',
    'cross_resolution_256_64.py', 'spectrogram_utils.py',
    'MECG-E/models/StudentNet.py', 'MECG-E/models/SSEMGNet.py',
    'config/config_spectrogram_v19_tt_mask.yaml',
    'config/student_16ch_1blk_causal_256_64.yaml',
    'model_weight/v19_tt_mask_v2_mix_nv0_weights.pth',
)
for relative in required:
    path = root / relative
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f'ERROR: required file missing/empty: {path}')
    print(f'[sha256] {hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}', flush=True)
loss_path = root / 'distill_loss_v3_256_64.py'
expected_loss = '99bfe549e3c0efca358126594435ebcf634950ed2d40fa14813c86b70089abd0'
if hashlib.sha256(loss_path.read_bytes()).hexdigest() != expected_loss:
    raise SystemExit('ERROR: original three-term loss is not the audited mask-fixed version; do not overwrite it blindly')
expected_teacher = '7b1f97a88449e2d9312d679846ea11f9108622e51e3d1be326463bcf2d139d04'
teacher_path = root / 'model_weight/v19_tt_mask_v2_mix_nv0_weights.pth'
if hashlib.sha256(teacher_path.read_bytes()).hexdigest() != expected_teacher:
    raise SystemExit('ERROR: Teacher content differs from the v2 bestval checkpoint used in the comparison')
cfgs = []
for filename in ('config_spectrogram_v19_tt_mask.yaml', 'student_16ch_1blk_causal_256_64.yaml'):
    with (root / 'config' / filename).open(encoding='utf-8-sig') as stream:
        cfgs.append(yaml.safe_load(stream))
teacher, student = cfgs
for cfg, grid in ((teacher, (512, 128, 512)), (student, (256, 64, 256))):
    h = cfg['model']
    if tuple(h[k] for k in ('n_fft', 'hop_size', 'win_size')) != grid:
        raise SystemExit(f'ERROR: wrong STFT grid; expected {grid}')
    if h.get('norm', False) is not False or h.get('fea') != 'pha' or h.get('sampling_rate') != 1000:
        raise SystemExit('ERROR: this experiment requires norm=False, fea=pha and fs=1000')
if (not student['model'].get('causal') or student['model'].get('dense_channel') != 16
        or student['model'].get('num_tscblocks') != 1):
    raise SystemExit('ERROR: expected causal 16-channel / 1-block Student')
for key, expected in dict(edepth=3, mdepth=3, pdepth=3, compress_factor=.5,
                          beta=1.0, mask_hp_cut_hz=150.0, use_gn=True,
                          gn_groups=8, causal_time_gn_groups=1).items():
    if student['model'].get(key) != expected:
        raise SystemExit(f'ERROR: Student {key} differs from the comparison experiment')
if student.get('ablation', {}).get('use_phase') is not True or student.get('ablation', {}).get('use_mrstft') is not True:
    raise SystemExit('ERROR: Student phase decoder and MR-STFT GT must stay enabled')
expected_metrics = dict(sampling_rate=1000, n_fft=256, hop_size=32, win_size=128, segment_sec=1)
if student.get('metrics') != expected_metrics:
    raise SystemExit('ERROR: evaluation feature settings differ from the comparison experiment')
if student['model'].get('loss_weights') != {'time': .25, 'com': .40, 'con': .25, 'mr': .08}:
    raise SystemExit('ERROR: Student GT weights differ from the comparison experiment')
if smoke == '0':
    for split in ('train', 'valid'):
        for kind in ('clean', 'noisy'):
            if not (data / split / kind).is_dir():
                raise SystemExit(f'ERROR: missing data directory: {data / split / kind}')
if not torch.cuda.is_available():
    raise SystemExit('ERROR: CUDA unavailable; training has not started (no CPU fallback)')
torch.empty(1, device='cuda:0')
print(f'[preflight] torch={torch.__version__} device={torch.cuda.get_device_name(0)}', flush=True)
print(f"[preflight] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}", flush=True)
print(f"[preflight] GT={student['model']['loss_weights']} | metrics={student.get('metrics')}", flush=True)
PY

if [[ "$SMOKE_ONLY" == 1 ]]; then
    exec "$PYTHON" -u test_timeKD_256_64_deterministic.py --project-root "$PROJ" --device cuda \
        --teacher-smoke --teacher-weights "$TEACHER_WEIGHTS"
fi

exec "$PYTHON" -u pipeline_distill_crossarch_v6_256_64_timeKD.py \
    --teacher_config "$TEACHER_CFG" \
    --teacher_weights "$TEACHER_WEIGHTS" \
    --student_config "$STUDENT_CFG" \
    --data_root "$DATA_ROOT" \
    --batch_size 4 --accum 4 \
    --num_workers 0 --epochs 150 --schedule_epochs 150 --patience 20 --seed "$SEED" \
    --w_feature 0 --w_response 1 --kd_weight_mode annealed --anneal_schedule linear \
    --w_resp_mask 3.5 --w_resp_mag 5.0 --w_resp_pha 0.0 --w_resp_com 2.3 \
    --w_resp_time "$W_TIME" \
    --model_save "$MODEL_SAVE" \
    --log_csv "$LOG_CSV" \
    "${RESUME_ARGS[@]}"
