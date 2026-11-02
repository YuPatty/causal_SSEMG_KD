"""FP32, batch-1 offline inference benchmark; no training and no test scoring.

Compare LiteEMG and causal 512/128, 256/128, 256/64, 128/32.
--suite all also runs the existing Teacher/FCN/MSEMG/SDEMG128 benchmarks
sequentially and writes one nine-model comparison table and CSV.
Uses actual checkpoints by default. Reports convolution/linear MACs and 2*MACs
FLOPs, explicitly excluding normalization, activation, bias, FFT and elementwise
ops. These counts do not reproduce an unspecified legacy profiling convention.
"""
from __future__ import annotations
import argparse
import csv
import subprocess
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--weights-dir", type=Path)
    p.add_argument("--weights-128-32", type=Path, help="Override 128/32 best weights only")
    p.add_argument("--suite", choices=("students", "all"), default="students")
    p.add_argument("--device", choices=("cpu", "cuda", "both"), default="both")
    p.add_argument("--seconds", type=float, default=10.)
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--cpu-runs", type=int, default=30)
    p.add_argument("--gpu-runs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--random-weights", action="store_true", help="Explicit architecture-only benchmark; never silently replace a missing checkpoint")
    p.add_argument("--allow-local-cpu", action="store_true", help="For a local workstation only; do not use this override on an HPC login node")
    return p.parse_args()


# Reject HPC login-node invocation before importing Torch.
if __name__ == "__main__":
    CLI = arguments()
    if not os.environ.get("SLURM_JOB_ID") and not (CLI.allow_local_cpu and CLI.device == "cpu"):
        raise SystemExit("Use sbatch on the cluster. Local CPU only: --device cpu --allow-local-cpu")
    if platform.node().lower().startswith(("cbi-lgn", "lgn", "login")):
        raise SystemExit("Refusing model work on an HPC login node")

import torch
from torch import nn


CASES = (
    ("LiteEMG_noncausal_512_128", "student_16ch_1blk.yaml", "student_student_16ch_1blk_respKD.pth"),
    ("Causal_512_128", "student_16ch_1blk_causal.yaml", "student_16ch_1blk_respKD_causal.pth"),
    ("Causal_256_128", "student_16ch_1blk_causal_256_128.yaml", "student_16ch_1blk_respKD_causal_256_128_teacher_v2_maskfix.pth"),
    ("Causal_256_64", "student_16ch_1blk_causal_256_64.yaml", "student_16ch_1blk_respKD_causal_256_64_teacher_v2_maskfix_dev.pth"),
    ("Causal_128_32", "student_16ch_1blk_causal_128_32.yaml", "student_16ch_1blk_respKD_causal_128_32_teacher_v2_maskfix_seed42.pth"),
)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_weights(model, path, allow_legacy_noncausal_prefix=False):
    raw = path.read_bytes()
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError(f"Expected best-weight state_dict, not resume state: {path}")
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    renamed = False
    if any(k.startswith("TSConv.") for k in state) and hasattr(model, "TFConv"):
        if not allow_legacy_noncausal_prefix or getattr(model, "causal", False):
            raise ValueError("Legacy TSConv keys are accepted only for the verified noncausal baseline")
        if any(k.startswith("TFConv.") for k in state):
            raise ValueError("Mixed TSConv/TFConv checkpoint keys")
        # The old noncausal implementation names the same block list TSConv.
        # Rename keys in memory only, then require an otherwise STRICT load.
        state = {("TFConv." + k[7:]) if k.startswith("TSConv.") else k: v for k, v in state.items()}
        renamed = True
    model.load_state_dict(state, strict=True)
    return hashlib.sha256(raw).hexdigest(), renamed


class InferenceOnly(nn.Module):
    """The actual pha inference path; excludes all GT/loss/auxiliary modules."""
    def __init__(self, student, cfg):
        super().__init__()
        if cfg["model"].get("fea") != "pha" or cfg["model"].get("norm", False) not in (False, None):
            raise ValueError("This benchmark matches the current pha/norm=False configurations only")
        if student.phase_decoder is None:
            raise ValueError("Expected the phase decoder used in the reported experiments")
        self.encoder = student.dense_encoder
        self.blocks = student.TFConv if hasattr(student, "TFConv") else student.TSConv
        self.mask = student.mask_decoder
        self.phase = student.phase_decoder

    def forward(self, spec):
        x = spec.permute(0, 1, 3, 2).contiguous()  # [B,2,T,F], known mag/phase format
        features = self.encoder(x)
        for block in self.blocks:
            features = block(features)
        mag = x[:, 0] * self.mask(features).squeeze(1)
        phase = self.phase(features).squeeze(1)
        return mag.transpose(1, 2).contiguous(), phase.transpose(1, 2).contiguous()


def convolution_macs(module, x, y):
    if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        return x.shape[0] * math.prod(x.shape[2:]) * module.weight.numel()
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return y.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
    if isinstance(module, nn.Linear):
        return y.numel() * module.in_features
    raise TypeError(type(module))


def count_main_ops(graph, spec):
    layers, handles = [], []
    types = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear)
    def make_hook(name):
        def hook(module, inputs, output):
            layers.append(dict(name=name, type=type(module).__name__,
                               input_shape=list(inputs[0].shape), output_shape=list(output.shape),
                               macs=int(convolution_macs(module, inputs[0], output))))
        return hook
    try:
        for name, module in graph.named_modules():
            if isinstance(module, types):
                handles.append(module.register_forward_hook(make_hook(name)))
        with torch.inference_mode():
            graph(spec)
    finally:
        for handle in handles:
            handle.remove()
    return dict(conv_linear_macs=sum(r["macs"] for r in layers), layers=layers)


def quantile(values, q):
    values = sorted(values)
    at = (len(values) - 1) * q
    lo = int(at)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (at - lo)


def measure(fn, device, warmup, runs, seconds):
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        sync()
        milliseconds = []
        for _ in range(runs):
            sync()
            start = time.perf_counter()
            fn()
            sync()
            milliseconds.append((time.perf_counter() - start) * 1000)
    median = statistics.median(milliseconds)
    return dict(median_ms=median, p95_ms=quantile(milliseconds, .95), min_ms=min(milliseconds),
                runs=runs, warmup=warmup, all_runs_ms=milliseconds,
                rtf=median / (seconds * 1000), audio_seconds=seconds)


def cpu_name():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def main(args):
    import yaml
    if (not math.isfinite(args.seconds) or args.seconds <= 0 or args.cpu_threads < 1
            or args.warmup < 1 or min(args.cpu_runs, args.gpu_runs) < 2):
        raise ValueError("Invalid duration, threads, warmup, or run counts")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite a previous result: {args.output}")
    if args.device in ("cuda", "both") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; requested GPU benchmark will not fall back to CPU")
    root = args.project_root.resolve()
    weights_dir = (args.weights_dir or root / "model_weight").resolve()
    sys.path.insert(0, str(root))
    source_path = root / "MECG-E/models/StudentNet.py"
    Student = load_module("efficiency_student", source_path).StudentSSEMGNet
    from spectrogram_utils import mag_pha_stft, mag_pha_istft
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    paths = [Path(__file__).resolve(), source_path, root / "spectrogram_utils.py"] + [root / "config" / c[1] for c in CASES]
    initial_hashes = {str(p): file_hash(p) for p in paths}
    # Check all inputs before starting measurements. No silent missing-weight fallback.
    if not args.random_weights:
        for _, _, weight in CASES:
            p = student_weight_path(args, weights_dir, weight)
            if not p.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {p}; explicitly use --random-weights only for architecture profiling")
    results = dict(environment=dict(torch=str(torch.__version__), python=platform.python_version(),
        platform=platform.platform(), host=platform.node(), cpu=cpu_name(),
        cpu_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), job_id=os.environ.get("SLURM_JOB_ID")),
        protocol=dict(dtype="float32", batch_size=1, seconds=args.seconds, seed=args.seed,
        random_weights=args.random_weights, timer="perf_counter wall time; synchronize before/after each GPU run",
        tf32=False, cudnn_benchmark=False, input="one identical seeded synthetic noisy waveform per model",
        timing_scope="device-resident inputs/outputs; excludes disk I/O and host-device transfers",
        flop_scope="Conv/ConvTranspose/Linear only; 1 MAC = 2 FLOPs; no bias/norm/activation/elementwise/FFT",
        streaming_latency="NOT measured; this is offline clip processing time; hop is only update interval"),
        source_sha256=initial_hashes, cases=[])
    print("[environment] " + json.dumps(results["environment"], ensure_ascii=False), flush=True)
    print("[protocol] " + json.dumps(results["protocol"], ensure_ascii=False), flush=True)
    devices = [torch.device(d) for d in (["cpu", "cuda:0"] if args.device == "both" else ["cuda:0" if args.device == "cuda" else "cpu"])]
    checkpoint_hashes = {}
    for label, cfg_name, weight_name in CASES:
        cfg = yaml.safe_load((root / "config" / cfg_name).read_text(encoding="utf-8-sig"))
        validate_student_grid(label, cfg)
        h = cfg["model"]
        if int(h.get("sampling_rate", 1000)) != 1000:
            raise ValueError("Expected the 1-kHz sEMG experiment configurations")
        n = int(round(1000 * args.seconds))
        if n <= int(h["n_fft"]):
            raise ValueError("Duration must exceed every model's FFT size")
        actual_seconds = n / 1000
        waveform = .03 * torch.randn(1, n, generator=torch.Generator().manual_seed(args.seed))
        student = Student(cfg).eval()
        checkpoint_sha = None
        legacy_rename = False
        checkpoint = student_weight_path(args, weights_dir, weight_name)
        if not args.random_weights:
            checkpoint_sha, legacy_rename = load_weights(student, checkpoint, allow_legacy_noncausal_prefix=not bool(getattr(student, "causal", False)))
            checkpoint_hashes[str(checkpoint)] = checkpoint_sha
        student.requires_grad_(False)
        total_params = sum(p.numel() for p in student.parameters())
        graph = InferenceOnly(student, cfg).eval()
        active_params = sum(p.numel() for p in graph.parameters())
        def to_spec(wav):
            mag, pha, _ = mag_pha_stft(wav, h["n_fft"], h["hop_size"], h["win_size"], h["compress_factor"])
            return torch.stack((mag, pha), 1).contiguous()
        def to_wav(mag, pha):
            return mag_pha_istft(mag, pha, h["n_fft"], h["hop_size"], h["win_size"], h["compress_factor"])
        with torch.inference_mode():
            cpu_spec = to_spec(waveform)
        counted = count_main_ops(graph, cpu_spec)
        case = dict(label=label, config=cfg_name, grid=[h[k] for k in ("n_fft", "hop_size", "win_size")],
                    causal=bool(getattr(student, "causal", False)), spec_shape=list(cpu_spec.shape),
                    weights_path=str(checkpoint) if not args.random_weights else None,
                    weights_sha256=checkpoint_sha, strict_load=not args.random_weights,
                    legacy_noncausal_TSConv_to_TFConv_in_memory=legacy_rename,
                    inference_parameters=active_params, full_training_parameters=total_params,
                    auxiliary_parameters_excluded=total_params-active_params,
                    conv_linear_gmacs=counted["conv_linear_macs"] / 1e9,
                    conv_linear_gflops=counted["conv_linear_macs"] * 2 / 1e9,
                    conv_linear_gflops_per_audio_second=counted["conv_linear_macs"] * 2 / 1e9 / actual_seconds,
                    layer_counts=counted["layers"], timing={})
        for device in devices:
            graph.to(device)
            wav = waveform.to(device)
            with torch.inference_mode():
                spec = to_spec(wav)
                enhanced = to_wav(*graph(spec))
                if not torch.isfinite(enhanced).all():
                    raise RuntimeError(f"Nonfinite inference output: {label}, {device}")
            runs = args.cpu_runs if device.type == "cpu" else args.gpu_runs
            net_time = measure(lambda: graph(spec), device, args.warmup, runs, actual_seconds)
            pipeline_time = measure(lambda: to_wav(*graph(to_spec(wav))), device, args.warmup, runs, actual_seconds)
            case["timing"][device.type] = dict(network_only=net_time, stft_network_istft=pipeline_time)
        results["cases"].append(case)
        public = {k: v for k, v in case.items() if k not in ("layer_counts", "timing")}
        public["timing"] = {dev: {scope: {k: v for k, v in stats.items() if k != "all_runs_ms"}
                                  for scope, stats in scopes.items()} for dev, scopes in case["timing"].items()}
        print("[CASE] " + json.dumps(public, ensure_ascii=False), flush=True)
        del graph, student
    for path, digest in {**initial_hashes, **checkpoint_hashes}.items():
        if file_hash(path) != digest:
            raise RuntimeError(f"Source/checkpoint changed during benchmark: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, allow_nan=False)
    print("\n===== EFFICIENCY SUMMARY: batch=1, FP32, offline clip =====")
    print(f"{'Model':29s} {'Params':>8s} {'ConvGFLOPs':>11s} {'CPU net ms':>11s} {'CPU pipe ms':>12s} {'GPU net ms':>11s} {'GPU pipe ms':>12s}")
    for row in results["cases"]:
        def ms(dev, scope):
            return f"{row['timing'][dev][scope]['median_ms']:.3f}" if dev in row["timing"] else "n/a"
        print(f"{row['label']:29s} {row['inference_parameters']:8d} {row['conv_linear_gflops']:11.6f} {ms('cpu','network_only'):>11s} {ms('cpu','stft_network_istft'):>12s} {ms('cuda','network_only'):>11s} {ms('cuda','stft_network_istft'):>12s}")
    print("ConvGFLOPs excludes bias/norm/activation/FFT; pipe = STFT + network + iSTFT.")
    print("Times are per offline clip, NOT streaming/algorithmic delay. Params exclude the training-only complex decoder.")
    print("[BENCHMARK PASS] sources/checkpoints unchanged; no training; result=" + str(args.output), flush=True)


# These functions are embedded in the delivered benchmark_liteemg_causal.py.
SUITE_TEACHER_SHA = '7b1f97a88449e2d9312d679846ea11f9108622e51e3d1be326463bcf2d139d04'
SUITE_BASELINE_SHAS = {
    'fcn_baseline.pth': '31137da876c18f1b0944b34f075b25a6c895968de92cd23aa70f594a93aa2519',
    'msemg_baseline.pth': 'df39b6c644dd2d49dff83955bd73a1764b7480b2c0afae9b5b5e363c47a1bed7',
    'sdemg_baseline_feats128.pth': '68538f9e2274578e67b781fb23eb852a28349d32312b165cdc9834141d8abd51',
}


def student_weight_path(args, directory, filename):
    if filename == CASES[-1][2] and args.weights_128_32 is not None:
        return args.weights_128_32.resolve()
    return (directory / filename).resolve()


def validate_student_grid(label, cfg):
    grid = (512, 128, 512) if label.startswith('LiteEMG') else (
        int(label.split('_')[1]), int(label.split('_')[2]), int(label.split('_')[1]))
    h = cfg['model']
    if tuple(h[k] for k in ('n_fft', 'hop_size', 'win_size')) != grid:
        raise ValueError(f'{label}: config grid differs from {grid}')
    if bool(h.get('causal', False)) != label.startswith('Causal_'):
        raise ValueError(f'{label}: config causal flag mismatch')


def build_suite_commands(args, root, paths):
    common = ['--project-root', str(root), '--device', args.device,
              '--seconds', str(args.seconds), '--cpu-threads', str(args.cpu_threads),
              '--warmup', str(args.warmup), '--cpu-runs', str(args.cpu_runs),
              '--gpu-runs', str(args.gpu_runs), '--seed', str(args.seed)]
    weights = (args.weights_dir or root / 'model_weight').resolve()
    student = [sys.executable, '-B', '-u', str(Path(__file__).resolve()),
               *common, '--suite', 'students', '--weights-dir', str(weights),
               '--output', str(paths['students'])]
    if args.weights_128_32 is not None:
        student += ['--weights-128-32', str(args.weights_128_32.resolve())]
    teacher = [sys.executable, '-B', '-u', str(root / 'benchmark_teacher_v2.py'),
               *common, '--weights', str(weights / 'v19_tt_mask_v2_mix_nv0_weights.pth'),
               '--output', str(paths['teacher'])]
    baselines = [sys.executable, '-B', '-u', str(root / 'benchmark_baselines.py'),
                 *common, '--models', 'fcn', 'msemg', 'sdemg128',
                 '--student-helper', str(Path(__file__).resolve()),
                 '--teacher-helper', str(root / 'benchmark_teacher_v2.py'),
                 '--fcn-weights', str(weights / 'fcn_baseline.pth'),
                 '--msemg-weights', str(weights / 'msemg_baseline.pth'),
                 '--sdemg128-weights', str(weights / 'sdemg_baseline_feats128.pth'),
                 '--output', str(paths['baselines'])]
    return [('students', student), ('teacher', teacher), ('baselines', baselines)]


def merge_suite_results(args, documents):
    """Normalize measured child results; never substitute historical timings."""
    student, teacher, baselines = (documents[k] for k in ('students', 'teacher', 'baselines'))
    if teacher.get('status') != 'complete' or baselines.get('status') != 'complete':
        raise ValueError('A child benchmark is incomplete; no final comparison table')
    if {r['label'] for r in student['cases']} != {c[0] for c in CASES} or len(student['cases']) != 5:
        raise ValueError('Expected exactly the five LiteEMG/Causal Student cases')
    if {r['model'] for r in baselines['cases']} != {'FCN', 'MSEMG', 'SDEMG128'} or len(baselines['cases']) != 3:
        raise ValueError('Expected FCN/MSEMG/SDEMG128 only')
    if teacher['label'] != 'Teacher_v2_512_128':
        raise ValueError('Unexpected Teacher result')
    if teacher['weights_sha256'] != SUITE_TEACHER_SHA:
        raise ValueError('Teacher identity differs from the previous comparison')
    identity = {k:student['environment'][k] for k in ('host', 'torch', 'python', 'cpu', 'cpu_threads')}
    for group, doc in documents.items():
        if {k:doc['environment'][k] for k in identity} != identity:
            raise ValueError(f'{group}: environment differs from other measurements')
        protocol = doc['protocol']
        if protocol['batch_size'] != 1 or protocol['dtype'] != 'float32':
            raise ValueError(f'{group}: FP32/batch=1 protocol mismatch')
        if abs(protocol['seconds'] - round(args.seconds * 1000) / 1000) > 1e-9:
            raise ValueError(f'{group}: input duration mismatch')
    rows = []
    devices = ('cpu', 'cuda') if args.device == 'both' else (args.device,)

    def measured(timing, device):
        record = timing[device]
        runs = args.cpu_runs if device == 'cpu' else args.gpu_runs
        if record['warmup'] != args.warmup or record['runs'] != runs:
            raise ValueError('Timing repeat/warmup mismatch')
        value = float(record['median_ms'])
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid measured median')
        if abs(record['audio_seconds'] - round(args.seconds * 1000) / 1000) > 1e-9:
            raise ValueError('Timing input duration mismatch')
        return value

    def row(label, grid, params, conv, scan):
        if not all(math.isfinite(float(x)) and float(x) >= 0 for x in (conv, scan)):
            raise ValueError('Invalid complexity values')
        return dict(model=label, stft_window_hop=grid, parameters=int(params),
                    conv_linear_gflops=float(conv), scan_gflops_estimate=float(scan),
                    cpu_network_ms=None, cpu_complete_ms=None,
                    gpu_network_ms=None, gpu_complete_ms=None)

    def add_spectral(target, timing):
        for device in devices:
            key = 'gpu' if device == 'cuda' else 'cpu'
            record = timing[device]
            target[key + '_network_ms'] = measured({device:record['network_only']}, device)
            target[key + '_complete_ms'] = measured({device:record['stft_network_istft']}, device)

    t = row('Teacher v2', '512/128', teacher['inference_parameters'],
            teacher['complexity']['conv_linear_gflops'], teacher['complexity']['selective_scan_gflops_estimate'])
    add_spectral(t, teacher['timing'])
    rows.append(t)
    by_baseline = {r['model']:r for r in baselines['cases']}
    for label, filename in (('FCN', 'fcn_baseline.pth'), ('MSEMG', 'msemg_baseline.pth'),
                            ('SDEMG128', 'sdemg_baseline_feats128.pth')):
        source = by_baseline[label]
        if source['weights_sha256'] != SUITE_BASELINE_SHAS[filename]:
            raise ValueError(f'{label}: checkpoint differs from previous benchmark')
        if label == 'SDEMG128' and (source['sdemg_settings']['feats'] != 128 or
                source['sdemg_settings']['denoise_steps'] != 50 or source['complexity']['denoiser_calls'] != 50):
            raise ValueError('Expected SDEMG feats128 with all 50 diffusion steps')
        target = row('SDEMG128 (50 steps)' if label == 'SDEMG128' else label, 'waveform', source['params'],
                     source['complexity']['conv_linear_gflops'], source['complexity']['selective_scan_gflops_estimate'])
        for device in devices:
            target[('gpu' if device == 'cuda' else 'cpu') + '_complete_ms'] = measured(source['timing'], device)
        rows.append(target)
    by_student = {r['label']:r for r in student['cases']}
    for label, _, _ in CASES:
        source = by_student[label]
        target = row(label, f"{source['grid'][0]}/{source['grid'][1]}", source['inference_parameters'],
                     source['conv_linear_gflops'], 0.)
        if not source['strict_load']:
            raise ValueError('Expected actual strict-loaded Student weights')
        add_spectral(target, source['timing'])
        rows.append(target)
    return rows


def print_suite_summary(rows, json_path, csv_path):
    print('\n===== ALL-MODEL EFFICIENCY SUMMARY: FP32, batch=1, offline clip =====', flush=True)
    print('Model                         STFT        Params  ConvLinearGFLOPs  ScanGFLOPs_est  CPU net ms  CPU complete ms  GPU net ms  GPU complete ms')
    for row in rows:
        def ms(key):
            return f"{row[key]:.3f}" if row[key] is not None else '--'
        print(f"{row['model']:29s} {row['stft_window_hop']:9s} {row['parameters']:8d} "
              f"{row['conv_linear_gflops']:17.6f} {row['scan_gflops_estimate']:15.6f} "
              f"{ms('cpu_network_ms'):>11s} {ms('cpu_complete_ms'):>16s} "
              f"{ms('gpu_network_ms'):>11s} {ms('gpu_complete_ms'):>16s}")
    print('Spectral complete = STFT + network + iSTFT; native waveform complete = full native inference path.')
    print('Conv/Linear MAC*2 only; scan estimates are separate. SDEMG128 uses all 50 sampling steps.')
    print('Offline clip times exclude I/O/transfers; not streaming, buffering, or physical end-to-end latency.')
    print(f'[ALL-MODEL BENCHMARK PASS] no training; sources/weights unchanged; JSON={json_path}; CSV={csv_path}', flush=True)


def run_all(args):
    import yaml
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('--suite all requires a real Slurm job; use --suite students for local CPU QA')
    if args.random_weights:
        raise ValueError('All-model comparison requires actual checkpoint weights')
    if (not math.isfinite(args.seconds) or args.seconds <= .512 or args.cpu_threads < 1
            or args.warmup < 1 or min(args.cpu_runs, args.gpu_runs) < 2):
        raise ValueError('Invalid suite timing settings')
    if args.device in ('both', 'cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    if args.output.suffix.lower() != '.json':
        raise ValueError('--output must have a .json extension')
    root = args.project_root.resolve()
    output = args.output.resolve()
    csv_path = output.with_suffix('.csv')
    parts = {key:output.with_name(output.stem + '_' + key + '.json')
             for key in ('students', 'teacher', 'baselines')}
    for path in (output, csv_path, *parts.values()):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite a previous suite output: {path}')
    weights = (args.weights_dir or root / 'model_weight').resolve()
    required = [Path(__file__).resolve(), root / 'benchmark_liteemg_causal.py',
                root / 'benchmark_teacher_v2.py', root / 'benchmark_baselines.py',
                root / 'benchmark_fcn_model.py', root / 'benchmark_msemg_model_source.py',
                root / 'spectrogram_utils.py', root / 'MECG-E/models/StudentNet.py',
                root / 'MECG-E/models/SSEMGNet.py', root / 'config/config_spectrogram_v19_tt_mask.yaml',
                weights / 'v19_tt_mask_v2_mix_nv0_weights.pth']
    required += [root / 'benchmark_sdemg_sources' / name
                 for name in ('deep_filter_model.py', 'ddpm_1d.py', 'utils.py', 'default.yaml')]
    for _, cfg, filename in CASES:
        required += [root / 'config' / cfg, student_weight_path(args, weights, filename)]
    required += [weights / filename for filename in SUITE_BASELINE_SHAS]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'Missing/empty suite input: {path}')
    hashes = {str(path):file_hash(path) for path in required}
    if hashes[str(weights / 'v19_tt_mask_v2_mix_nv0_weights.pth')] != SUITE_TEACHER_SHA:
        raise ValueError('Teacher checkpoint differs from the previous v2 benchmark')
    for filename, expected in SUITE_BASELINE_SHAS.items():
        if hashes[str(weights / filename)] != expected:
            raise ValueError(f'Baseline checkpoint differs from the previous benchmark: {filename}')
    for label, cfg_name, _ in CASES:
        validate_student_grid(label, yaml.safe_load((root / 'config' / cfg_name).read_text(encoding='utf-8-sig')))
    output.parent.mkdir(parents=True, exist_ok=True)
    print('[suite] 9 models; sequential subprocesses on one allocation, same timing protocol; no training', flush=True)
    print('[suite] Teacher and SDEMG128 CPU measurements can take tens of minutes', flush=True)
    for stage, command in build_suite_commands(args, root, parts):
        print(f'[suite stage] {stage}: starting; result={parts[stage]}', flush=True)
        try:
            subprocess.run(command, cwd=root, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f'{stage} benchmark failed (exit={exc.returncode}); completed child JSONs retained; no final all-model PASS') from exc
        print(f'[suite stage] {stage}: complete', flush=True)
    for path, expected in hashes.items():
        if file_hash(path) != expected:
            raise RuntimeError(f'Suite source/checkpoint changed: {path}')
    documents = {key:json.loads(path.read_text(encoding='utf-8')) for key, path in parts.items()}
    rows = merge_suite_results(args, documents)
    with csv_path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(status='complete', environment=documents['students']['environment'],
                  protocol=dict(seconds=round(args.seconds * 1000) / 1000, sampling_rate=1000,
                                batch_size=1, dtype='float32', cpu_threads=args.cpu_threads,
                                warmup=args.warmup, cpu_runs=args.cpu_runs, gpu_runs=args.gpu_runs,
                                seed=args.seed, execution='sequential subprocesses; one allocation',
                                scope='offline waveform inference; no disk I/O/host-device transfers/streaming delay'),
                  source_checkpoint_sha256=hashes,
                  part_result_paths={key:str(path) for key, path in parts.items()}, rows=rows)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print_suite_summary(rows, output, csv_path)


if __name__ == "__main__":
    run_all(CLI) if CLI.suite == "all" else main(CLI)
