"""Standalone v2 Teacher benchmark; reuses the completed Student benchmark helpers.

Same 10-s FP32 batch-1 timing scopes. Main Conv/Linear FLOPs include Mamba
projections and depthwise convolution analytically, even under fused kernels.
Selective-scan cost is a separately labeled VMamba-style estimate, not a
measurement of complete inference FLOPs. No training/model/checkpoint writes.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--weights", type=Path)
    p.add_argument("--device", choices=("both", "cuda", "cpu"), default="both")
    p.add_argument("--seconds", type=float, default=10.)
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--cpu-runs", type=int, default=30)
    p.add_argument("--gpu-runs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


if __name__ == "__main__":
    CLI = arguments()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Use sbatch on the cluster; no Teacher benchmark on the login node")

import torch
from torch import nn


class TeacherInference(nn.Module):
    def __init__(self, teacher, cfg):
        super().__init__()
        h = cfg["model"]
        if h.get("fea") != "pha" or h.get("norm", False) not in (False, None):
            raise ValueError("Expected the current pha, norm=False Teacher")
        if teacher.phase_decoder is None:
            raise ValueError("Expected the Teacher phase decoder")
        self.encoder = teacher.dense_encoder
        self.blocks = teacher.TSMamba
        self.mask = teacher.mask_decoder
        self.phase = teacher.phase_decoder

    def forward(self, spec):
        x = spec.permute(0, 1, 3, 2).contiguous()
        features = self.encoder(x)
        for block in self.blocks:
            features = block(features)
        mag = x[:, 0] * self.mask(features).squeeze(1)
        phase = self.phase(features).squeeze(1)
        return mag.transpose(1, 2).contiguous(), phase.transpose(1, 2).contiguous()


def mamba_main_cost(module, shape):
    """Cost for a full-sequence real-valued Mamba-1 call; no state-cache step."""
    if len(shape) != 3:
        raise ValueError("Expected Mamba input [B,L,D]")
    b, length, d = map(int, shape)
    inner, state, rank, kernel = map(int, (module.d_inner, module.d_state, module.dt_rank, module.d_conv))
    expected = {"in_proj": (2 * inner, d), "x_proj": (rank + 2 * state, inner),
                "dt_proj": (inner, rank), "out_proj": (d, inner)}
    for key, dimensions in expected.items():
        if tuple(getattr(module, key).weight.shape) != dimensions:
            raise ValueError(f"Unsupported Mamba layout: {key}")
    if tuple(module.A_log.shape) != (inner, state) or module.A_log.is_complex():
        raise ValueError("Only real-valued Mamba-1 is supported")
    if tuple(module.conv1d.weight.shape) != (inner, 1, kernel) or module.conv1d.groups != inner:
        raise ValueError("Unexpected Mamba depthwise convolution")
    per_token = dict(in_proj=d * 2 * inner, depthwise_conv=inner * kernel,
                     x_proj=inner * (rank + 2 * state), dt_proj=rank * inner,
                     out_proj=inner * d)
    macs = b * length * sum(per_token.values())
    # Conventional selective-scan estimate used by VMamba's authors:
    # 9*B*L*D_inner*N, plus D-skip and z-gate terms. Kept SEPARATE from
    # the explicit 2-FLOPs-per-MAC projection/convolution count.
    skip = b * length * inner if module.D is not None else 0
    gate = b * length * inner
    scan_estimate = 9 * b * length * inner * state + skip + gate
    return dict(input_shape=list(shape), d_inner=inner, d_state=state, dt_rank=rank,
                d_conv=kernel, projection_conv_macs=macs, macs_per_token=per_token,
                selective_scan_cost_estimate=scan_estimate)


def count_teacher_ops(graph, spec, mamba_class, conventional_counter):
    """Count the outer Mamba module once; skip ALL its child-module hooks."""
    mixers = {name: mod for name, mod in graph.named_modules() if isinstance(mod, mamba_class)}
    if not mixers:
        raise ValueError("No Mamba modules were found; refusing an incomplete Teacher count")
    rows, handles = [], []
    def mixer_hook(name):
        def hook(module, inputs, output):
            if tuple(output.shape) != tuple(inputs[0].shape):
                raise ValueError("Unexpected Mamba output shape")
            rows.append(dict(name=name, type="Mamba-1", **mamba_main_cost(module, inputs[0].shape)))
        return hook
    def ordinary_hook(name):
        def hook(module, inputs, output):
            rows.append(dict(name=name, type=type(module).__name__,
                             input_shape=list(inputs[0].shape), output_shape=list(output.shape),
                             projection_conv_macs=int(conventional_counter(module, inputs[0], output)),
                             selective_scan_cost_estimate=0))
        return hook
    conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear)
    try:
        for name, module in graph.named_modules():
            if name in mixers:
                handles.append(module.register_forward_hook(mixer_hook(name)))
            elif isinstance(module, conv_types) and not any(not p or name.startswith(p + ".") for p in mixers):
                handles.append(module.register_forward_hook(ordinary_hook(name)))
        with torch.inference_mode():
            graph(spec)
    finally:
        for handle in handles:
            handle.remove()
    if sum(r["type"] == "Mamba-1" for r in rows) != len(mixers):
        raise ValueError("Some Mamba mixers were skipped or called more than once; inspect this architecture before profiling")
    return dict(conv_linear_gflops=sum(r["projection_conv_macs"] for r in rows) * 2 / 1e9,
                selective_scan_gflops_estimate=sum(r["selective_scan_cost_estimate"] for r in rows) / 1e9,
                mamba_calls=len(mixers), layer_counts=rows)


def main(args):
    import yaml
    root = args.project_root.resolve()
    sys.path.insert(0, str(root))
    from benchmark_liteemg_causal import load_module, load_weights, measure, file_hash, convolution_macs, cpu_name
    from spectrogram_utils import mag_pha_stft, mag_pha_istft
    if (not math.isfinite(args.seconds) or args.seconds <= 0 or args.cpu_threads < 1
            or args.warmup < 1 or min(args.cpu_runs, args.gpu_runs) < 2):
        raise ValueError("Invalid duration/threads/warmup/runs")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    if args.device in ("both", "cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; requested GPU measurement will not fall back to CPU")
    teacher_path = root / "MECG-E/models/SSEMGNet.py"
    config_path = root / "config/config_spectrogram_v19_tt_mask.yaml"
    weights_path = (args.weights or root / "model_weight/v19_tt_mask_v2_mix_nv0_weights.pth").resolve()
    sources = [Path(__file__).resolve(), root / "benchmark_liteemg_causal.py", teacher_path,
               config_path, root / "spectrogram_utils.py"]
    hashes = {str(p): file_hash(p) for p in sources}
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    module = load_module("efficiency_teacher", teacher_path)
    if args.device in ("cpu", "both") and not callable(getattr(module, "set_model_device_mode", None)):
        raise RuntimeError("Teacher source lacks set_model_device_mode CPU compatibility; do not overwrite the model. Use --device cuda for GPU-only profiling, or inspect this source version first.")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    h = cfg["model"]
    if tuple(h[k] for k in ("n_fft", "hop_size", "win_size")) != (512, 128, 512) or int(h.get("sampling_rate", 1000)) != 1000:
        raise ValueError("Expected the current Teacher 512/128/512 at 1kHz")
    n = round(1000 * args.seconds)
    if n <= 512:
        raise ValueError("Duration must exceed FFT length")
    teacher = module.SSEMGNet(cfg).eval()
    checkpoint_sha, _ = load_weights(teacher, weights_path)
    expected_teacher_sha = "7b1f97a88449e2d9312d679846ea11f9108622e51e3d1be326463bcf2d139d04"
    if checkpoint_sha != expected_teacher_sha:
        raise ValueError("Teacher weights differ from the previously verified v2 best-validation checkpoint")
    teacher.requires_grad_(False)
    original_state = {k:v.detach().cpu().clone() for k,v in teacher.state_dict().items()}
    waveform = .03 * torch.randn(1, n, generator=torch.Generator().manual_seed(args.seed))
    result = dict(label="Teacher_v2_512_128", weights=str(weights_path), weights_sha256=checkpoint_sha,
                  strict_load=True, grid=[512,128,512], full_training_parameters=sum(p.numel() for p in teacher.parameters()),
                  source_sha256=hashes, environment=dict(torch=str(torch.__version__), python=platform.python_version(),
                  host=platform.node(), cpu=cpu_name(), cpu_threads=torch.get_num_threads(),
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  mamba_ssm=importlib.metadata.version("mamba-ssm"), job_id=os.environ.get("SLURM_JOB_ID")),
                  protocol=dict(seconds=n/1000, batch_size=1, dtype="float32", seed=args.seed,
                  timing="same measure() helper as Student; device-resident input; wall-clock + CUDA synchronization",
                  flop_scope="Conv/Linear incl. Mamba projections/conv, MAC*2; no norm/bias/activation/FFT",
                  scan_scope="SEPARATE VMamba-style scan estimate: 9*B*L*D_inner*N + D-skip + z-gate",
                  scan_source="https://github.com/MzeroMiko/VMamba/blob/main/classification/models/csms6s.py",
                  total_inference_flops="NOT claimed; two scopes are reported separately",
                  streaming_latency="NOT measured; offline clip processing time"), timing={}, status="running")
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print("[environment] " + json.dumps(result["environment"]), flush=True)
    print("[protocol] " + json.dumps(result["protocol"]), flush=True)
    devices = [torch.device(d) for d in (["cuda:0", "cpu"] if args.device == "both" else ["cuda:0" if args.device == "cuda" else "cpu"])]
    reference = None
    try:
        for device in devices:
            print(f"[stage] Teacher {device}: preparing backend; no training", flush=True)
            teacher.to(device)
            setter = getattr(module, "set_model_device_mode", None)
            if callable(setter):
                setter(teacher, device)
            teacher.requires_grad_(False)
            mixers = [m for m in teacher.modules() if isinstance(m, module.Mamba)]
            if not mixers or any(bool(m.use_fast_path) != (device.type == "cuda") for m in mixers):
                raise ValueError("Mamba backend mode does not match the requested device")
            graph = TeacherInference(teacher, cfg).eval()
            wav = waveform.to(device)
            def to_spec(w):
                mag, pha, _ = mag_pha_stft(w, 512,128,512,h["compress_factor"])
                return torch.stack((mag, pha), 1).contiguous()
            def to_wav(mag, pha):
                return mag_pha_istft(mag,pha,512,128,512,h["compress_factor"])
            with torch.inference_mode():
                spec = to_spec(wav)
                prediction = graph(spec)
                enhanced = to_wav(*prediction)
                if not torch.isfinite(enhanced).all():
                    raise ValueError("Non-finite Teacher output")
                # Optional source-provided inference oracle, outside the timer.
                if callable(getattr(teacher, "infer_spectrogram", None)):
                    oracle = teacher.infer_spectrogram(spec)
                    torch.testing.assert_close(prediction[0], oracle["mag_g_FT"], rtol=1e-4, atol=1e-6)
                    delta = prediction[1] - oracle["pha_g"]
                    if float((1 - delta.cos()).abs().max()) > 1e-5:
                        raise ValueError("Teacher phase wrapper differs from its inference oracle")
                else:
                    teacher(spec,spec)  # QA only; never part of timed inference
                    torch.testing.assert_close(prediction[0], teacher.last_outputs["mag_g_FT"], rtol=1e-4, atol=1e-6)
                    delta = prediction[1] - teacher.last_outputs["pha_g"]
                    if float((1-delta.cos()).abs().max()) > 1e-5:
                        raise ValueError("Teacher phase wrapper differs from full-forward oracle")
            params = sum(p.numel() for p in graph.parameters())
            result["inference_parameters"] = params
            result["auxiliary_parameters_excluded"] = result["full_training_parameters"] - params
            if "complexity" not in result:
                result["complexity"] = count_teacher_ops(graph, spec, module.Mamba, convolution_macs)
                print("[complexity] " + json.dumps({k:v for k,v in result["complexity"].items() if k != "layer_counts"}), flush=True)
            runs = args.gpu_runs if device.type == "cuda" else args.cpu_runs
            print(f"[stage] {device} network-only: warmup={args.warmup}, runs={runs}; CPU reference scan may take minutes", flush=True)
            net = measure(lambda: graph(spec), device, args.warmup, runs, n/1000)
            print(f"[stage] {device} STFT+network+iSTFT: warmup={args.warmup}, runs={runs}", flush=True)
            pipe = measure(lambda: to_wav(*graph(to_spec(wav))), device, args.warmup, runs, n/1000)
            result["timing"][device.type] = dict(network_only=net, stft_network_istft=pipe,
                 backend="CUDA Mamba fast path requested" if device.type == "cuda" else "PyTorch reference selective scan + plain RMSNorm",
                 mamba_use_fast_path=sorted(set(bool(m.use_fast_path) for m in mixers)),
                 causal_conv1d_kernel_available=bool(getattr(getattr(module,"_mamba_simple_module",None),"causal_conv1d_fn",None)))
            for name, value in teacher.state_dict().items():
                if not torch.equal(value.detach().cpu(), original_state[name]):
                    raise ValueError(f"In-memory Teacher parameter/buffer changed: {name}")
            if reference is not None:
                result["cpu_gpu_waveform_relative_l2_informational"] = float((enhanced.cpu()-reference).norm()/reference.norm().clamp_min(1e-20))
            reference = enhanced.detach().cpu()
            print(f"[DEVICE RESULT] {device.type} | params={params} | net={net['median_ms']:.3f} ms | pipe={pipe['median_ms']:.3f} ms", flush=True)
        for path, digest in {**hashes, str(weights_path): checkpoint_sha}.items():
            if file_hash(path) != digest:
                raise RuntimeError(f"Source/checkpoint changed during measurement: {path}")
        result["status"] = "complete"
        save()
    except Exception as exc:
        result["status"] = "failed_or_partial"
        result["error"] = str(exc)
        if not args.output.exists():
            save()
        raise
    print("\n===== TEACHER EFFICIENCY SUMMARY: batch=1, FP32, offline clip =====")
    print("Model                 Params  ConvLinearGFLOPs  ScanGFLOPs_est  CPU net ms  CPU pipe ms  GPU net ms  GPU pipe ms")
    def ms(device,scope):
        return f"{result['timing'][device][scope]['median_ms']:.3f}" if device in result["timing"] else "n/a"
    print(f"Teacher_v2_512_128 {result['inference_parameters']:9d} {result['complexity']['conv_linear_gflops']:17.6f} {result['complexity']['selective_scan_gflops_estimate']:15.6f} {ms('cpu','network_only'):>11s} {ms('cpu','stft_network_istft'):>12s} {ms('cuda','network_only'):>11s} {ms('cuda','stft_network_istft'):>12s}")
    print("Scan cost is an analytical estimate shown separately; ConvLinear includes fused Mamba projections/depthwise conv.")
    print("CPU reference-scan implementation and GPU fused implementation are recorded; time is not streaming delay.")
    print("[TEACHER BENCHMARK PASS] no training, sources/weights unchanged; result=" + str(args.output), flush=True)


if __name__ == "__main__":
    main(CLI)
