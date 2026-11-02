# Reproducing the recorded experiments

This update adds the 128/32 experiment without changing the existing 512/128,
256/128 or 256/64 training implementations. Training, quality evaluation,
offline timing and streaming timing are separate operations.

## Environment and external inputs

The recorded compute-node environment was Linux, Python 3.11.15,
PyTorch 2.3.1+cu121, NumPy 1.26.4 and librosa 0.11.0. Teacher runs used
`mamba-ssm==2.2.2`. Follow the main README for CUDA/Mamba installation.
`requirements.txt` is a dependency list, not a complete machine environment lock.
For Student-only inference, `pip install -r requirements-student.txt` avoids
installing Teacher-only CUDA extensions. A fresh installation and full dataset
evaluation have not been rerun on every platform.

Obtain NinaPro DB2 and the MIT-BIH Normal Sinus Rhythm Database through their
respective providers, then use the documented preprocessing and split.
Datasets and pretrained checkpoints are **not included** in this repository;
there is no automatic public checkpoint download. Train the required models or
obtain the exact checkpoints from the project author. The pinned Teacher
preflight requires the recorded v2 checkpoint, not an arbitrary newly trained
Teacher. The available hashes are in
[`recorded_results/checkpoints.sha256`](recorded_results/checkpoints.sha256).
An unchecked or differently trained checkpoint does not reproduce the recorded run.

Place checkpoints in `model_weight/`. Use the best-validation `.pth` for
inference; use `.resume_state.pth` only to continue training. Preserve the full
state, CSV, optimizer/scheduler/RNG state and experiment-specific output names.
Do not run dev and normal writers against the same experiment concurrently.

For per-SNR evaluation, obtain or verify `test_snr_labels.json` against the exact
waveform dataset ordering (`collect_waveform_pairs`). The existing
`build_snr_labels.py` reads legacy `.pt` data and is not a verified replacement
for the waveform-order label file.

## 128/32 training

Teacher grid: 512/128/512. Student grid: 128/32/128 (FFT/hop/window),
sampling rate 1000 Hz. Student: 16 channels, one TFConv block, causal path.
Seed 42, batch size 4, accumulation 4, workers 0, 150 epochs,
150-epoch linear alpha schedule, patience 20, AdamW learning rate 3e-4.

`L_total = (1-alpha) L_GT + alpha L_KD`, with
`L_KD = 3.5 L_mask + 5.0 L_mag + 2.3 L_complex` and
`alpha = max(0, 1 - epoch/150)` for the one-based epoch used by the pipeline.
Feature, response-phase and Time-KD weights are zero in this main experiment.
GT weights remain time=0.25, complex=0.40, consistency=0.25, MR-STFT=0.08.

The supplied shell scripts are **NCHC Slurm examples**. Their account,
partition, excluded node, log paths, Python and dataset defaults are specific
to the recorded cluster, not universal installation defaults. Create `logs/`
before submitting; Slurm opens log files before the script body runs.

On the original cluster, from the project root:

```bash
mkdir -p logs model_weight results
bash -n run_student_causal_KD_128_32.sh
sbatch --export=ALL,SMOKE_ONLY=1,RESUME=0 run_student_causal_KD_128_32.sh
# After the preflight job passes:
sbatch --export=ALL,SMOKE_ONLY=0,RESUME=0 run_student_causal_KD_128_32.sh
sbatch --export=ALL,SMOKE_ONLY=0,RESUME=0 run_student_causal_noKD_128_32.sh
```

The two training jobs use different outputs. Fresh runs refuse existing output
files; continuation requires all three original best/state/CSV files:

```bash
sbatch --export=ALL,SMOKE_ONLY=0,RESUME=1 run_student_causal_KD_128_32.sh
```

For another Slurm installation, export `PROJ`, `DATA_ROOT`,
`SSEMG_CONDA_PREFIX` and optionally `PYTHON`, and override the account,
partition, excluded nodes and log destinations via `sbatch` or edit the example
directives. No model computation should run on an HPC login node.

Outside Slurm, the equivalent KD Python command is:

```bash
mkdir -p model_weight logs results
python -u pipeline_distill_crossarch_v6_128_32.py \
  --teacher_config config/config_spectrogram_v19_tt_mask.yaml \
  --teacher_weights model_weight/v19_tt_mask_v2_mix_nv0_weights.pth \
  --student_config config/student_16ch_1blk_causal_128_32.yaml \
  --data_root /path/to/processed \
  --batch_size 4 --accum 4 --num_workers 0 \
  --epochs 150 --schedule_epochs 150 --patience 20 --seed 42 \
  --w_feature 0 --w_response 1 --kd_weight_mode annealed --anneal_schedule linear \
  --w_resp_mask 3.5 --w_resp_mag 5 --w_resp_pha 0 --w_resp_com 2.3 \
  --model_save model_weight/student_12832_KD_seed42.pth \
  --log_csv logs/student_12832_KD_seed42.csv
```

The NoKD experiment uses the existing dynamic-STFT branch of
`pipeline_distill_crossarch_v6_256_64.py`, with the 128/32 YAML, `--no_kd`,
the same batch/accumulation/schedule/seed, and separate model/CSV paths.
The filename does not fix the NoKD input grid at 256/64. Its NoKD branch
constructs no Teacher and optimizes only GT loss.
The raw Python CLI lacks the shell wrapper's output lock/refusal checks;
choose new paths for fresh experiments.

## 128/32 inference

The shared inference entry point reads FFT/hop/window dimensions from the YAML:

```bash
mkdir -p results
python inference_student_patched_v6_256_64.py \
  --config config/student_16ch_1blk_causal_128_32.yaml \
  --weights model_weight/student_16ch_1blk_respKD_causal_128_32_teacher_v2_maskfix_seed42.pth \
  --dataset /path/to/processed --batch 64 --index 1786 \
  --csv-out results/student_KD_12832_seed42_metrics.csv \
  --exp-alias student_KD_12832_seed42
```

Add `--snr_labels /path/to/test_snr_labels.json` only after ordering is verified.
On the original cluster the supplied `run_infer_student_KD_128_32.sh` and
`run_infer_student_noKD_128_32.sh` require this label file and CUDA, perform a
strict model preflight, and refuse to overwrite a previous result alias.
The shared Python inference supports CPU; the Slurm wrapper intentionally
requires GPU. Keep metrics fixed at 1000 Hz / FFT 256 / hop 32 / window 128 /
1-second segments across input grids.

## Historical Time-KD experiment

The two `*_256_64_timeKD.py` files and corresponding training shell script
reproduce the separate waveform-distillation variant, adding
`0.25 L_timeKD` to the three response losses. Its recorded SNRimp was 20.6001 dB,
below the 20.7771 dB three-term run; it is not the main 128/32 configuration.
Use `run_smoke_timeKD_256_64_deterministic.sh` for the recorded scoped
gradient-equivalence smoke check. Its determinism controls apply to the test,
not production training; CUDA reflection padding in other backward paths has
known nondeterministic behavior in the recorded environment.

## Interpretation

These are single-run observations. Identical seed numbers do not ensure
identical Student initialization because KD constructs the Teacher first.
For a controlled ablation, pair the same initial Student state and data order
across KD/NoKD and repeat multiple seeds. Use validation to select settings.
Cross-grid composite GT losses are not direct substitutes for final waveform
SNRimp. Validation `val_mf` and inference `RMSE_MF` must be checked by their
definitions: the recorded validation path uses median frequency, whereas
inference `RMSE_MF`/`RMSE_MeanF` denotes mean frequency.
