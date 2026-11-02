# Validation of this repository update (2026-11-02)

This update packages existing experiment implementations and recorded results.
It does not represent a new full training run or a new H100 benchmark.

- Existing tracked Python/YAML files are byte-identical to base commit
  `15452765bcc844437719737735d9c9ffdd4c8904` in the committed Git tree.
- All 18 added Python files parse successfully. All nine added shell scripts
  pass `bash -n` and reject execution outside a Slurm allocation before model
  computation (the small smoke wrapper delegates to the guarded training script).
- The 32 copied source/wrapper files match the publication manifest. Production
  Python and source snapshot bytes are preserved; shell changes allow exported
  path/interpreter overrides and select the validated deterministic smoke test.
- The 128/32 NoKD synthetic check passes forward, GT loss, backward and a
  disposable optimizer step; no experiment weights or CSV are written.
- `test_timeKD_256_64_deterministic.py` reports 34 tests, OK, one skipped
  (the optional real Teacher CUDA check). It covers waveform alignment,
  zero-Time-KD equivalence, gradients and resume/output guards.
- The five-Student benchmark passes a one-second CPU integration check with
  real existing checkpoints, one warmup and two timed repeats. The 128/32 QA
  explicitly uses an available NoKD checkpoint to validate that architecture;
  it is not the final KD quality checkpoint or a replacement for job 431195.
- Markdown local links/code fences and added documentation/shell/result-file
  whitespace checks pass. Historical Python source formatting is preserved.

Local runtime: Windows, Python 3.13.2, PyTorch 2.7.1+cpu, one Torch/OMP thread.
These local checks do not validate Linux CUDA/Mamba, installation on a fresh
Python 3.11 environment, full-dataset SNRimp, or target wearable timing.
The documented H100 measurements are the previously completed cluster jobs.

Local synthetic checks can be repeated from the repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python check_student_noKD_128_32.py \
  --project-root . --mode smoke --allow-local-cpu
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python test_timeKD_256_64_deterministic.py \
  --project-root .
```

The explicit local-CPU option is for a workstation, not an HPC login node.
Submit the supplied Slurm scripts for compute-node checks involving Teacher
weights or full timing. Their hardware-specific directives must be adapted
on other clusters.
