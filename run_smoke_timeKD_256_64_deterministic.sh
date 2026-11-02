#!/bin/bash
#SBATCH -A MST115015
#SBATCH -p dev
#SBATCH -J smoke_timeKD_det
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=hgpn17
#SBATCH -t 00:20:00
#SBATCH -o /work/taes10056/SSEMG-Net/logs/slurm_%j_smoke_timeKD_deterministic.out
#SBATCH -e /work/taes10056/SSEMG-Net/logs/slurm_%j_smoke_timeKD_deterministic.err

# Full smoke rerun. Determinism is scoped to the zero-time gradient unit test.
# No production source/checkpoint/CSV is modified, and no tolerance is relaxed.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'ERROR: submit with sbatch; no Torch tests on a login node.' >&2
    exit 1
fi
PROJ=${PROJ:-/work/taes10056/SSEMG-Net}
DATA_ROOT=${DATA_ROOT:-/work/taes10056/semg_data_40sub/processed}
export CONDA_PREFIX=${SSEMG_CONDA_PREFIX:-/home/taes10056/.conda/envs/ssemg_env_311}
PYTHON=${PYTHON:-$CONDA_PREFIX/bin/python}
SMOKE_ONLY=1
cd "$PROJ"
[[ -x "$PYTHON" ]] || { echo "ERROR: missing Python: $PYTHON" >&2; exit 1; }
module load cuda 2>/dev/null || true
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
NV_LIBS=""
if [[ -d "$CONDA_PREFIX/lib/python3.11/site-packages/nvidia" ]]; then
    NV_LIBS=$(find "$CONDA_PREFIX/lib/python3.11/site-packages/nvidia" -type d -name lib -print | tr '\n' ':')
fi
export LD_LIBRARY_PATH="${NV_LIBS}${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
echo "Job: $SLURM_JOB_ID Host: $(hostname); full smoke with scoped deterministic gradient comparison"
echo "[smoke-only] No experiment weights or training CSV will be written."

"$PYTHON" - "$PROJ" "$DATA_ROOT" "$SMOKE_ONLY" <<'PY'
import hashlib
import os
import sys
from pathlib import Path
import torch
import yaml

root, data, smoke = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
required = (
    'test_timeKD_256_64_deterministic.py',
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


exec "$PYTHON" -u test_timeKD_256_64_deterministic.py --project-root "$PROJ" \
    --device cuda --teacher-smoke \
    --teacher-weights "$PROJ/model_weight/v19_tt_mask_v2_mix_nv0_weights.pth"
