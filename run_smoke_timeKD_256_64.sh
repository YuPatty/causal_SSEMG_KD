#!/bin/bash
#SBATCH -A MST115015
#SBATCH -p dev
#SBATCH -J smoke_timeKD
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=hgpn17
#SBATCH -t 00:20:00
#SBATCH -o /work/taes10056/SSEMG-Net/logs/slurm_%j_smoke_timeKD.out
#SBATCH -e /work/taes10056/SSEMG-Net/logs/slurm_%j_smoke_timeKD.err
set -euo pipefail
export SMOKE_ONLY=1
PROJ=${PROJ:-/work/taes10056/SSEMG-Net}
exec bash "$PROJ/run_student_causal_KD_256_64_timeKD.sh"
