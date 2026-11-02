# Offline efficiency and streaming reference measurements

These are different timing protocols; do not mix their units or interpret the
offline table as streaming latency. Rounded transcriptions of completed job
outputs are under [`recorded_results/`](recorded_results/). They are not the
original detailed JSON artifacts or new measurements from this GitHub update.

## Offline: all nine models, job 431195

One allocation on hgpn01; Xeon Platinum 8480+, one CPU thread, H100 80 GB,
PyTorch 2.3.1+cu121, FP32, batch 1, 10-second synthetic waveform at 1000 Hz.
Warmup 5; CPU 30 repeats, GPU 50 repeats; median wall time with CUDA
synchronization. Device-resident inputs; no loading, I/O, acquisition or
host/device transfers. Spectral pipeline = STFT + network + iSTFT; waveform
baselines execute their full native waveform path.

| Model | Grid | Params | Conv/Linear GFLOPs | Scan estimate GFLOPs | CPU net ms | CPU complete ms | GPU net ms | GPU complete ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Teacher v2 | 512/128 | 2,258,825 | 51.513370 | 6.047138 | 10521.658 | 10508.192 | 10.641 | 11.032 |
| FCN | waveform | 137,801 | 0.963151 | 0 | — | 18.989 | — | 0.383 |
| MSEMG | waveform | 279,937 | 5.491200 | 0.373760 | — | 629.309 | — | 1.409 |
| SDEMG128, 50 steps | waveform | 4,925,313 | 3902.864384 | 0 | — | 39286.219 | — | 322.068 |
| LiteEMG noncausal | 512/128 | 46,153 | 1.190761 | 0 | 25.788 | 27.076 | 1.346 | 1.906 |
| Causal Student | 512/128 | 46,153 | 1.190761 | 0 | 55.588 | 55.718 | 1.983 | 2.547 |
| Causal Student | 256/128 | 46,025 | 0.596479 | 0 | 28.005 | 28.837 | 1.983 | 2.554 |
| Causal Student | 256/64 | 46,025 | 1.185408 | 0 | 56.098 | 58.176 | 1.979 | 2.563 |
| Causal Student | 128/32 | 45,961 | 1.185985 | 0 | 59.783 | 59.331 | 1.986 | 2.551 |

GFLOPs are **per 10-second input**, with MAC counted as two operations.
Conv/Linear counts exclude bias, normalization, activation, FFT and other
elementwise operations. Mamba projection/conv cost is included; the
[VMamba-style scan estimate](https://github.com/MzeroMiko/VMamba/blob/main/classification/models/csms6s.py)
is separate. Neither column is a claim of total inference FLOPs.
Inference parameter counts exclude the Student/Teacher training-only complex
decoder. SDEMG counts registered parameters (including original unused final
stream blocks), and all 50 reverse-diffusion steps; executed Conv/Linear
parameters were 4,072,577. CPU Mamba uses a reference scan, GPU uses the fused
implementation, so their speed ratio is implementation-specific.
Network and complete scopes are timed independently; complete medians can
be slightly below network medians because of measurement variability.

The all-model runner requires the Teacher and baseline helpers and preserved
source snapshots now included in the repository, plus all nine checkpoints.
MSEMG and SDEMG license notices are retained in `LICENSE_MSEMG.txt`,
`NOTICE_MSEMG.txt` and `benchmark_sdemg_sources/LICENSE`.
It never substitutes random weights. On the original cluster:

```bash
mkdir -p logs results
bash -n run_benchmark_liteemg_causal.sh
sbatch run_benchmark_liteemg_causal.sh
```

This uses `--suite all` and separate sequential subprocesses to avoid mixing
Teacher/Student backend settings. The Python default `--suite students`
remains available for the five spectral Students. A failed stage does not
produce a final all-model PASS. Outputs use the new Slurm job ID and refuse
existing destinations. Detailed historical artifacts remain on the cluster:
`results/efficiency_all_models_j431195.json`, its `.csv`, and the
`_students.json`, `_teacher.json`, `_baselines.json` files.

## Streaming reference: four Student grids, job 419538

Same server model and precision, batch 1, one CPU thread. A seeded 19,968-sample
synthetic waveform is processed per hop. The implementation recomputes a
bounded history (49 STFT frames); it is not an optimized per-layer cache.
It validates future-frame perturbations, delayed waveform prefix agreement
and streaming/offline-prefix parity before timing. `center=True` STFT is
retained and emission is delayed until the needed samples have arrived.

| Grid | Device | Hop ms | Buffer end..start ms | Net median ms | Complete median ms | Complete p95 ms | Max ms | Over hop / measured hops |
|---|---|---:|---|---:|---:|---:|---:|---|
| 512/128 | CPU | 128 | 384–512 | 33.291 | 33.606 | 34.022 | 35.153 | 0/102 |
| 512/128 | GPU | 128 | 384–512 | 2.511 | 2.918 | 2.956 | 2.976 | 0/102 |
| 256/128 | CPU | 128 | 128–256 | 18.552 | 18.765 | 18.935 | 19.317 | 0/103 |
| 256/128 | GPU | 128 | 128–256 | 2.479 | 2.902 | 2.927 | 2.956 | 0/103 |
| 256/64 | CPU | 64 | 192–256 | 18.539 | 18.806 | 18.929 | 19.322 | 0/258 |
| 256/64 | GPU | 64 | 192–256 | 2.484 | 2.921 | 2.957 | 2.989 | 0/258 |
| 128/32 | CPU | 32 | 96–128 | 10.916 | 11.312 | 12.099 | 12.515 | 0/570 |
| 128/32 | GPU | 32 | 96–128 | 2.476 | 2.886 | 2.920 | 3.344 | 0/570 |

Buffer intervals are **derived** from this validated reference emission
schedule: `(N-H)/fs` through `N/fs`, using block-end through block-start
timestamps. They are not microphone wall-clock measurements or universal
minimum delays. Complete processing includes FFT, network, WOLA/iFFT and
Python buffer handling; it excludes acquisition, transfer, output-device
delay, file I/O and EOF flush. RTF in the detailed output is mean complete
hop time divided by hop interval, not median divided by hop.
P95 below hop in this run indicates processing headroom, not a hard real-time
guarantee or validation on a wearable target. Synthetic probes do not prove
causality for all possible inputs. Startup/EOF behavior is outside the
steady-state timing summary.

```bash
mkdir -p logs results
bash -n run_audit_causal_stream_4grids.sh
sbatch run_audit_causal_stream_4grids.sh
```

The reference audit's 128/32 checkpoint was the snapshot available at job
419538, during training. Final quality job 430975 used the later completed
checkpoint. Timing reflects the same architecture, but the old prefix audit
does not substitute for validating the final checkpoint; rerun the audit to
associate a new snapshot hash with it. Final quality is offline evaluation,
not a new streaming-quality measurement.

Teacher, LiteEMG noncausal, MSEMG and SDEMG have no validated native streamer
in this update. Do not assign them zero buffering or use their offline times
as streaming delays. A newer separate FCN delayed-stream comparison has only
local QA so far and is not included as a completed cluster result here.
