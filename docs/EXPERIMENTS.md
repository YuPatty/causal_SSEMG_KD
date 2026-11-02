# Recorded experiment provenance (updated 2026-11-02)

The README keeps the earlier results and adds the completed 128/32 experiments.
The CSVs here are **rounded console-summary transcriptions**, not original
per-sample predictions, raw training logs, uncertainty estimates or newly
generated benchmark outputs. No raw dataset or weights are redistributed.

| Record | Source | Scope |
|---|---|---|
| [Quality CSV](recorded_results/quality_recorded.csv) | Completed inference jobs, listed per row | Test-set Average; no multi-seed averaging |
| [Offline efficiency CSV](recorded_results/efficiency_j431195_summary.csv) | Completed job 431195, all-model PASS | Nine models, 10-second offline input |
| [Streaming CSV](recorded_results/streaming_j419538_summary.csv) | Completed job 419538, four-grid PASS | Per-hop steady state and derived buffering |
| [Known checkpoint hashes](recorded_results/checkpoints.sha256) | Retained checkpoint verification outputs | Partial manifest, not all files |
| [Source manifest](recorded_results/source_manifest.json) | This repository update | Package origin and exact byte hashes |

The 128/32 NoKD inference is job 426115, with SNRimp 20.6076 dB.
The final 128/32 response-KD inference is job 430975, with SNRimp 21.1378 dB.
Training jobs 416821 and 428616 completed the latter run in two allocations,
ending at epoch 150. Its quality checkpoint is
`student_16ch_1blk_respKD_causal_128_32_teacher_v2_maskfix_seed42.pth`.
This is distinct from the earlier in-training snapshot used in streaming
audit 419538; do not describe the old stream timing as a new final-checkpoint
quality audit.

Inference `RMSE_MF` and `RMSE_MeanF` name the same mean-frequency metric, so
only one is reported. PRD is intentionally omitted from the summary table.
Printed average RMSE must not be reconstructed by averaging per-sample CSV
RMSE without checking the original aggregation. Test-set averages and
per-SNR averages also use different aggregations for some metrics.

The individual KD-minus-NoKD SNRimp differences are 0.9875 dB (512/128),
0.4288 dB (256/128), 0.2400 dB (256/64) and 0.5302 dB (128/32).
These are not controlled paired-seed effect estimates. Earlier runs include
resume/concurrent-writer history, and Teacher initialization consumes RNG
before Student initialization. Keep those limitations when interpreting or
reporting the comparisons. The older Teacher/512-series seeds remain
unconfirmed; the recorded 256/128-series Students use seed 42.

The existing 512/256 production files are preserved in this update. New
128/32, Time-KD and measurement Python files are copied from the delivered
experiment packages. Shell wrappers retain original cluster defaults but now
accept exported path/interpreter overrides. Time-KD's shell smoke path selects
the scoped deterministic test that passed job 400198; production loss and
training behavior are unchanged. Snapshot source licenses are retained.

See [reproduction requirements](REPRODUCING.md) and
[measurement definitions](BENCHMARKS.md) for commands and limits.
