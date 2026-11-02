"""Four-grid Student causality and hop-processing audit; no training or test scoring.

The reference streamer preserves centered periodic-Hann STFT and weighted OLA.
It recomputes the finite network history on every hop (not an optimized layer-
cache implementation). Timings apply to this reference implementation only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--config-128-32', type=Path)
    p.add_argument('--weights-dir', type=Path)
    p.add_argument('--cases', nargs='+', choices=('512_128', '256_128', '256_64', '128_32'),
                   default=['512_128', '256_128', '256_64', '128_32'])
    p.add_argument('--device', choices=('cpu', 'cuda', 'both'), default='both')
    p.add_argument('--seconds', type=float, default=20.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--warmup-hops', type=int, default=5)
    p.add_argument('--cpu-threads', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--random-weights', action='store_true', help='Explicit architecture QA only; no silent missing-weight fallback')
    p.add_argument('--allow-local-cpu', action='store_true', help='Local workstation QA only, never an HPC login-node override')
    return p.parse_args()


# Run this guard before importing Torch.
if __name__ == '__main__':
    CLI = arguments()
    if not os.environ.get('SLURM_JOB_ID') and not (CLI.allow_local_cpu and CLI.device == 'cpu'):
        raise SystemExit('Use sbatch on the cluster. Local QA: --device cpu --allow-local-cpu')
    if platform.node().lower().startswith(('cbi-lgn', 'lgn', 'login')):
        raise SystemExit('Refusing to run a model audit on an HPC login node')

import torch
from torch import nn


WEIGHTS = {
    '512_128': 'student_16ch_1blk_respKD_causal.pth',
    '256_128': 'student_16ch_1blk_respKD_causal_256_128_teacher_v2_maskfix.pth',
    '128_32': 'student_16ch_1blk_respKD_causal_128_32_teacher_v2_maskfix_seed42.pth',
    '256_64': 'student_16ch_1blk_respKD_causal_256_64_teacher_v2_maskfix_dev.pth',
}

CONFIGS = {
    '512_128': 'student_16ch_1blk_causal.yaml',
    '256_128': 'student_16ch_1blk_causal_256_128.yaml',
    '256_64': 'student_16ch_1blk_causal_256_64.yaml',
    '128_32': 'student_16ch_1blk_causal_128_32.yaml',
}


def shared_samples(cases, seconds, sampling_rate=1000):
    """All cases receive identical samples, aligned to every selected hop."""
    alignment = math.lcm(*(int(case.split('_')[1]) for case in cases))
    return (int(round(seconds * sampling_rate)) // alignment) * alignment


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot_weights(model, path):
    """Take a stable in-memory snapshot; a running trainer may update the file later."""
    for attempt in range(5):
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            time.sleep(.2)
            continue
        try:
            state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
        except (RuntimeError, EOFError, OSError):
            if attempt == 4:
                raise
            time.sleep(.2)
            continue
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
            raise ValueError(f'Expected best-weight state_dict, not resume state: {path}')
        if all(k.startswith('module.') for k in state):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                    size_bytes=len(raw), mtime_ns=after.st_mtime_ns, strict_load=True,
                    note='In-memory snapshot; the trainer may legitimately update the source file later')
    raise RuntimeError(f'Checkpoint is changing during read; rerun this audit later: {path}')


class InferenceCore(nn.Module):
    """Same mag/phase inference path as inference_student_patched_v6_256_64.py."""
    def __init__(self, student, cfg):
        super().__init__()
        h = cfg['model']
        if h.get('fea') != 'pha' or h.get('norm', False) not in (False, None):
            raise ValueError('Expected pha inputs and norm=False')
        if not student.causal or student.phase_decoder is None:
            raise ValueError('Expected the causal Student with its inference phase decoder')
        self.encoder = student.dense_encoder
        self.blocks = student.TFConv
        self.mask = student.mask_decoder
        self.phase = student.phase_decoder

    def forward(self, spec):
        x = spec.permute(0, 1, 3, 2).contiguous()
        feat = self.encoder(x)
        for block in self.blocks:
            feat = block(feat)
        mag = x[:, 0] * self.mask(feat).squeeze(1)
        phase = self.phase(feat).squeeze(1)
        return mag.transpose(1, 2).contiguous(), phase.transpose(1, 2).contiguous()


def to_complex(mag, phase, compress):
    return torch.polar(mag.clamp_min(0).pow(1. / compress), phase)


def past_context(core):
    """Read the actual temporal padding, rather than assume a fixed RF size."""
    def dense_past(dense):
        total = 0
        for layer in dense.dense_block:
            wrapper = layer[0]
            if not hasattr(wrapper, 'pad_t'):
                raise ValueError('Dense block is not using a causal temporal convolution')
            conv = wrapper.conv
            expected = (conv.kernel_size[0] - 1) * conv.dilation[0]
            if wrapper.pad_t != expected or conv.padding[0] != 0 or conv.stride[0] != 1:
                raise ValueError('Unsupported temporal padding/stride')
            total += expected
        return total
    encoder = dense_past(core.encoder.dense_block)
    middle = 0
    for block in core.blocks:
        for conv in (block.time_conv1, block.time_conv2):
            if not conv.causal_time or conv.depthwise.padding != (0,) or conv.depthwise.stride != (1,):
                raise ValueError('TF block time convolution is not causal')
            middle += conv.left_pad
    mask = dense_past(core.mask.dense_block)
    phase = dense_past(core.phase.dense_block)
    return dict(encoder_past_frames=encoder, tf_past_frames=middle,
                mask_past_frames=mask, phase_past_frames=phase,
                past_frames=encoder + middle + max(mask, phase))


def difference(a, b):
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise RuntimeError(f'Invalid or nonfinite comparison: {a.shape} vs {b.shape}')
    return dict(max_abs=float((a - b).abs().max()),
                relative_l2=float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-12)))


def require_close(a, b, label):
    stats = difference(a, b)
    # Fixed before measurements; FP32 shape-dependent convolution roundoff is allowed.
    torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4, msg=label)
    return stats


def check_frame_causality(core, spec, compress, history):
    reference = to_complex(*core(spec), compress)
    results = []
    for cut in sorted(set((history + 3, spec.shape[-1] // 2, spec.shape[-1] - 5))):
        if not 1 <= cut < spec.shape[-1]:
            continue
        altered = spec.clone()
        altered[:, 0, :, cut:] = spec[:, 0, :, cut:].flip(-1) * .37 + .1
        altered[:, 1, :, cut:] = spec[:, 1, :, cut:].flip(-1) * -.71 + .2
        result = to_complex(*core(altered), compress)
        stats = require_close(result[..., :cut], reference[..., :cut], 'Future STFT frames changed earlier predictions')
        results.append(dict(cut_frame=cut, **stats))
    if len(results) < 2:
        raise ValueError('Need a longer waveform for multiple frame-causality probes')
    return results


class CenteredHannReferenceStream:
    """Accept H new waveform samples, retain bounded input/context/OLA state.

    No right-boundary padding or EOF flush. Emit only waveform samples that
    future frames cannot change. Negative-time STFT context uses the original
    reflect convention. Each hop recomputes up to (past_frames+1) network frames.
    """
    def __init__(self, core, h, history, device):
        self.core, self.n, self.hop = core, int(h['n_fft']), int(h['hop_size'])
        self.half, self.compress, self.history = self.n // 2, float(h['compress_factor']), history
        self.device = device
        self.window = torch.hann_window(self.n, device=device)
        self.offsets = torch.arange(self.n, device=device, dtype=torch.long)
        self.raw = torch.empty(1, 0, device=device)
        self.raw_start = self.received = self.frame_index = 0
        self.spec_history = None
        self.numerator = torch.zeros(1, self.n, device=device)
        self.denominator = torch.zeros(1, self.n, device=device)

    def push(self, chunk):
        if chunk.shape != (1, self.hop):
            raise ValueError('Expected exactly one new hop, batch=1')
        self.raw = torch.cat((self.raw, chunk), -1)
        self.received += self.hop
        output, frames = [], 0
        # First reflected frame needs sample n_fft/2, as well as its right half.
        while (self.frame_index * self.hop + self.half <= self.received
               and self.received > self.half):
            start = self.frame_index * self.hop - self.half
            positions = (self.offsets + start).abs() - self.raw_start
            frame = self.raw.index_select(-1, positions)
            fft = torch.fft.rfft(frame * self.window, n=self.n)
            spec = torch.stack((fft.abs().pow(self.compress), fft.angle()), 1).unsqueeze(-1)
            if self.spec_history is None:
                self.spec_history = spec
            else:
                self.spec_history = torch.cat((self.spec_history, spec), -1)
            self.spec_history = self.spec_history[..., -(self.history + 1):]
            mag, phase = self.core(self.spec_history)
            enhanced = to_complex(mag[..., -1], phase[..., -1], self.compress)
            synthesis = torch.fft.irfft(enhanced, n=self.n) * self.window
            self.numerator += synthesis
            self.denominator += self.window.square()
            # The next frame starts at start+hop, so this prefix is finalized.
            if start >= 0:
                block = self.numerator[:, :self.hop] / self.denominator[:, :self.hop].clamp_min(1e-12)
                output.append(block.clone())
            self.numerator = torch.cat((self.numerator[:, self.hop:], torch.zeros_like(chunk)), -1)
            self.denominator = torch.cat((self.denominator[:, self.hop:], torch.zeros_like(chunk)), -1)
            self.frame_index += 1
            frames += 1
        keep_from = max(0, self.frame_index * self.hop - self.half)
        trim = keep_from - self.raw_start
        self.raw = self.raw[:, trim:]
        self.raw_start = keep_from
        return output, frames


def quantile(values, q):
    x = sorted(values)
    at = (len(x) - 1) * q
    lo = int(at)
    return x[lo] + (x[min(lo + 1, len(x) - 1)] - x[lo]) * (at - lo)


def timing_summary(values, hop_ms):
    if len(values) < 10:
        raise ValueError('Too few steady-state timed hops; increase --seconds')
    return dict(median_ms=statistics.median(values), p95_ms=quantile(values, .95),
                max_ms=max(values), min_ms=min(values), measured_hops=len(values),
                observed_over_hop_count=sum(v > hop_ms for v in values),
                rtf=statistics.mean(values) / hop_ms)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_stream(core, h, wav, history, warmup_hops, timed):
    device, hop = wav.device, int(h['hop_size'])
    stream = CenteredHannReferenceStream(core, h, history, device)
    values, output, first_emission = [], [], None
    for start in range(0, wav.shape[-1], hop):
        chunk = wav[:, start:start + hop]
        if timed:
            sync(device)
            begin = time.perf_counter()
        blocks, frames = stream.push(chunk)
        if timed:
            sync(device)
            elapsed = (time.perf_counter() - begin) * 1000
            if frames == 1 and stream.frame_index >= history + 1 + warmup_hops and blocks:
                values.append(elapsed)
        if blocks and first_emission is None:
            first_emission = stream.received
        output.extend(blocks)
    if not output:
        raise RuntimeError('No waveform blocks emitted')
    return torch.cat(output, -1), values, first_emission


def check_waveform_lookahead(core, h, wav, to_spec, to_wav):
    hop, n = int(h['hop_size']), int(h['n_fft'])
    cut = (wav.shape[-1] // (2 * hop)) * hop
    other = wav.clone()
    other[..., cut:] = wav[..., cut:].flip(-1) * -.83 + .01
    a = to_wav(*core(to_spec(wav)))
    b = to_wav(*core(to_spec(other)))
    finalized_prefix = cut - (n - hop)
    stats = require_close(b[..., :finalized_prefix], a[..., :finalized_prefix],
                          'Future waveform changed the already-finalized delayed output')
    unshifted = difference(b[..., :cut], a[..., :cut])
    return dict(input_cut_sample=cut, stable_prefix_samples=finalized_prefix,
                committed_prefix_check=stats, unshifted_prefix_difference=unshifted,
                zero_lookahead_effect_observed=unshifted['max_abs'] > 2e-5,
                note='Finite synthetic probes, not a mathematical proof for all inputs')


def main(args):
    import yaml
    root = args.project_root.resolve()
    weights_dir = (args.weights_dir or root / 'model_weight').resolve()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite: {args.output}')
    if not math.isfinite(args.seconds) or args.seconds <= 0 or args.cpu_threads < 1 or args.warmup_hops < 1:
        raise ValueError('Invalid duration, thread count or warmup')
    if args.device in ('cuda', 'both') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback for requested GPU audit')
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    source = root / 'MECG-E/models/StudentNet.py'
    utils_path = root / 'spectrogram_utils.py'
    protected = {str(p): digest(p) for p in (source, utils_path, Path(__file__))}
    Student = load_module('stream_audit_student', source).StudentSSEMGNet
    utils = load_module('stream_audit_original_stft', utils_path)
    devices = ['cpu', 'cuda:0'] if args.device == 'both' else ['cuda:0' if args.device == 'cuda' else 'cpu']
    result = dict(environment=dict(torch=str(torch.__version__), python=platform.python_version(),
        host=platform.node(), cpu_threads=torch.get_num_threads(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), job_id=os.environ.get('SLURM_JOB_ID')),
        protocol=dict(batch_size=1, dtype='float32', seed=args.seed, random_weights=args.random_weights,
        input='The same seeded synthetic waveform for every selected grid; not the test set',
        implementation='Bounded-history recomputation per hop; NOT an optimized layer-cache streamer',
        timing_scope='Wall time + CUDA synchronization; device-resident hops; FFT + network + WOLA + Python buffer management',
        excludes='Acquisition, host/device transfer, OS scheduling jitter outside each measured call, I/O, model loading, EOF flush',
        tolerances=dict(atol=2e-5, rtol=2e-4),
        determinism='No backward; fixed seed, deterministic cuDNN, TF32 off; no universal bitwise determinism claim'), cases=[])
    print('[environment] ' + json.dumps(result['environment']), flush=True)
    print('[protocol] ' + json.dumps(result['protocol']), flush=True)
    samples = shared_samples(args.cases, args.seconds)
    waveform = .03 * torch.randn(1, samples, generator=torch.Generator().manual_seed(args.seed))
    print('[comparison] ' + json.dumps(dict(cases=args.cases, requested_seconds=args.seconds,
          identical_waveform_samples=samples, actual_seconds=samples / 1000,
          weights_dir=str(weights_dir))), flush=True)
    for case in args.cases:
        cfg_path = (args.config_128_32 if case == '128_32' and args.config_128_32 else
                    root / 'config' / CONFIGS[case]).resolve()
        protected[str(cfg_path)] = digest(cfg_path)
        cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8-sig'))
        h = cfg['model']
        n, hop, win = (int(h[k]) for k in ('n_fft', 'hop_size', 'win_size'))
        sr, compress = int(h['sampling_rate']), float(h['compress_factor'])
        if (n, hop) != tuple(map(int, case.split('_'))) or n != win or n % (2 * hop) != 0 or sr != 1000:
            raise ValueError('Reference streamer supports the current even, centered, full-window grids only')
        student = Student(cfg).eval().requires_grad_(False)
        weights = None if args.random_weights else snapshot_weights(student, weights_dir / WEIGHTS[case])
        core = InferenceCore(student, cfg).eval()
        context = past_context(core)
        history = context['past_frames']
        if samples // hop < 2 * (history + args.warmup_hops + 5):
            raise ValueError('Need more audio for causality and steady-state probes; use --seconds 20 or longer')
        row = dict(label=case, config=str(cfg_path), grid=[n, hop, win], checkpoint=weights,
                   inference_parameters=sum(p.numel() for p in core.parameters()), context=context,
                   requested_seconds=args.seconds, actual_seconds=samples / sr,
                   frame_center_stft_lookahead_ms=(n / 2) * 1000 / sr,
                   reference_buffering=dict(first_block_after_input_samples=n,
                       hop_ms=hop * 1000 / sr, block_end_timestamp_delay_ms=(n - hop) * 1000 / sr,
                       block_start_timestamp_delay_ms=n * 1000 / sr,
                       note='Continuous-time block-boundary convention; sample-index conventions can differ by 1 sample. Not measured wall-clock microphone latency.'), devices={})
        for device_name in devices:
            device = torch.device(device_name)
            core.to(device)
            wav = waveform.to(device)
            def to_spec(x):
                m, p, _ = utils.mag_pha_stft(x, n, hop, win, compress)
                return torch.stack((m, p), 1).contiguous()
            def to_wav(m, p):
                return utils.mag_pha_istft(m, p, n, hop, win, compress)
            print(f'[stage] {case} {device}: frame causality, waveform lookahead and bounded-hop parity', flush=True)
            with torch.inference_mode():
                spec = to_spec(wav)
                frame_check = check_frame_causality(core, spec, compress, history)
                wave_check = check_waveform_lookahead(core, h, wav, to_spec, to_wav)
                offline = to_wav(*core(spec))
                # Untimed full pass warms kernels and independently checks the streamer.
                streamed, _, first = run_stream(core, h, wav, history, args.warmup_hops, False)
                if first != n or streamed.shape[-1] != samples - (n - hop):
                    raise AssertionError('Unexpected startup/commit boundary')
                parity = require_close(streamed, offline[..., :streamed.shape[-1]], 'Streaming prefix differs from original offline waveform')
                print(f'[stage] {case} {device}: timed one-hop waveform processing', flush=True)
                timed_wave, values, _ = run_stream(core, h, wav, history, args.warmup_hops, True)
                require_close(timed_wave, streamed, 'Timed streamer differs from validated stream')
                pipe_time = timing_summary(values, hop * 1000 / sr)
                context_spec = spec[..., -(history + 1):].contiguous()
                for _ in range(args.warmup_hops):
                    core(context_spec)
                sync(device)
                net_values = []
                for _ in range(50 if device.type == 'cuda' else 30):
                    sync(device)
                    begin = time.perf_counter()
                    core(context_spec)
                    sync(device)
                    net_values.append((time.perf_counter() - begin) * 1000)
                net_time = timing_summary(net_values, hop * 1000 / sr)
            info = dict(frame_causality=frame_check, waveform_lookahead=wave_check,
                        streaming_offline_prefix_parity=parity, compared_samples=streamed.shape[-1],
                        network_context_recompute=net_time, complete_hop=pipe_time,
                        block_end_delay_plus_p95_compute_ms=row['reference_buffering']['block_end_timestamp_delay_ms'] + pipe_time['p95_ms'],
                        block_start_delay_plus_p95_compute_ms=row['reference_buffering']['block_start_timestamp_delay_ms'] + pipe_time['p95_ms'])
            row['devices'][device.type] = info
            print('[DEVICE AUDIT] ' + json.dumps(dict(case=case, device=device.type,
                  frame_causality_pass=True, streaming_prefix_parity=parity,
                  hop_ms=hop * 1000 / sr, complete_hop=pipe_time)), flush=True)
        result['cases'].append(row)
        del core, student
    for path, original_hash in protected.items():
        if digest(path) != original_hash:
            raise RuntimeError(f'Source/config changed during audit: {path}')
    result['source_sha256'] = protected
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print('\n===== CAUSAL STREAM FOUR-GRID SUMMARY: FP32, batch=1 =====')
    print('Grid    Device  Hop(ms)  Buffer(ms,end..start)  Net median(ms)  Pipe median(ms)  Pipe p95(ms)  Max(ms)  OverHop')
    for row in result['cases']:
        b = row['reference_buffering']
        for dev, info in row['devices'].items():
            net, pipe = info['network_context_recompute'], info['complete_hop']
            print(f"{row['label']:7s} {dev:6s} {b['hop_ms']:7.1f} {b['block_end_timestamp_delay_ms']:7.1f}..{b['block_start_timestamp_delay_ms']:<7.1f} "
                  f"{net['median_ms']:14.3f} {pipe['median_ms']:16.3f} {pipe['p95_ms']:13.3f} {pipe['max_ms']:8.3f} {pipe['observed_over_hop_count']:5d}/{pipe['measured_hops']}")
    print('Buffer is derived from the validated reference emission schedule, not microphone wall-clock measurement.')
    print('P95 < Hop is evidence of headroom in this run, NOT a hard real-time guarantee.')
    print('No optimized layer caches; no acquisition/transfer/EOF flush; synthetic probes do not prove all-input causality.')
    print('[FOUR-GRID STREAM AUDIT PASS] inference only; original sources/checkpoints not written; result=' + str(args.output), flush=True)


if __name__ == '__main__':
    main(CLI)
