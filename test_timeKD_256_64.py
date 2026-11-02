"""CPU regression checks; optional real checkpoint CUDA smoke on a compute node.

Local: python test_timeKD_256_64.py --project-root /path/to/project
Cluster: submit run_smoke_timeKD_256_64.sh (never run Torch on login nodes).
Tests write only disposable synthetic fixtures, never experiment outputs.
"""
import argparse
import ast
import contextlib
import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent)
parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
parser.add_argument('--teacher-smoke', action='store_true')
parser.add_argument('--teacher-weights', type=Path)
ARGS, TEST_ARGS = parser.parse_known_args()
ROOT = ARGS.project_root.resolve()
PACKAGE = Path(__file__).resolve().parent
if ARGS.device == 'cuda' and not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable; no CPU fallback')
if ARGS.teacher_smoke and ARGS.device != 'cuda':
    raise SystemExit('Real Teacher smoke requires --device cuda on a compute node')
DEVICE = torch.device(ARGS.device)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACKAGE))
from cross_resolution_256_64 import waveform_batch_to_spec
from distill_loss_v3_256_64 import (
    annealed_alpha, get_layer_curriculum_indices, get_noise_std,
    response_distill_loss_cross_resolution,
)
from distill_loss_v3_256_64_timeKD import (
    response_kd_with_time, student_waveform_with_grad, waveform_time_distill_loss,
)
from spectrogram_utils import mag_pha_stft, mag_pha_istft

with (ROOT / 'config/student_16ch_1blk_causal_256_64.yaml').open(encoding='utf-8-sig') as f:
    CFG = yaml.safe_load(f)
with (ROOT / 'config/config_spectrogram_v19_tt_mask.yaml').open(encoding='utf-8-sig') as f:
    TCFG = yaml.safe_load(f)
spec = importlib.util.spec_from_file_location('time_kd_test_student', ROOT / 'MECG-E/models/StudentNet.py')
student_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(student_module)
Student = student_module.StudentSSEMGNet
SOURCE = (PACKAGE / 'pipeline_distill_crossarch_v6_256_64_timeKD.py').read_text(encoding='utf-8-sig')
WEIGHTS = dict(mask=3.5, mag=5.0, pha=0.0, com=2.3, time=.25)


def student():
    torch.manual_seed(42)
    return Student(copy.deepcopy(CFG)).to(DEVICE)


class FakeTeacher(torch.nn.Module):
    """Synthetic stand-in ONLY for CPU training-loop/resume regression tests."""
    def __init__(self, cfg):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(.8))
        self.cfg = cfg

    def forward(self, clean_spec, noisy_spec):
        self.last_wavs = {'pred': mag_pha_istft(
            noisy_spec[:, 0] * self.gain, noisy_spec[:, 1], 512, 128, 512, .5,
        ).detach()}
        return torch.zeros((), device=self.gain.device)


class SyntheticDataset(torch.utils.data.Dataset):
    interrupt_after = None

    def __init__(self, root, split):
        self.split, self.visits = split, 0

    def __len__(self):
        return 3 if self.split == 'train' else 2

    def __getitem__(self, index):
        if self.split == 'train':
            if self.interrupt_after is not None and self.visits >= self.interrupt_after:
                raise RuntimeError('synthetic interruption')
            self.visits += 1
        t = torch.arange(1024, dtype=torch.float32)
        clean = .03 * torch.sin(t * (.19 + .01 * index))
        noisy = clean + .02 * torch.cos(t * .07 + index)
        return clean, noisy


class Progress:
    def __init__(self, iterable, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, **kwargs):
        pass


def pipeline_namespace(source=SOURCE, old=False):
    """Execute actual function AST without importing CUDA-only Mamba on CPU.

    The entire copied training loop, optimizer and resume code execute. Only
    dataset, Teacher, GPU selector, progress and val_mf helper are synthetic.
    This does NOT substitute for the optional real-Teacher CUDA smoke below.
    """
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not old:
        nodes += [n for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'TIME_KD_CSV_HEADER' for t in n.targets)]
    namespace = dict(
        __file__=str(PACKAGE / 'pipeline_distill_crossarch_v6_256_64_timeKD.py'),
        ROOT_DIR=str(ROOT), torch=torch, os=os, csv=csv, json=json, hashlib=hashlib,
        math=math, tempfile=tempfile, DataLoader=DataLoader, tqdm=Progress,
        WaveformPairDataset=SyntheticDataset, StudentSSEMGNet=Student, SSEMGNet=FakeTeacher,
        waveform_batch_to_spec=waveform_batch_to_spec,
        response_distill_loss_cross_resolution=response_distill_loss_cross_resolution,
        response_kd_with_time=response_kd_with_time,
        annealed_alpha=annealed_alpha, get_layer_curriculum_indices=get_layer_curriculum_indices,
        get_noise_std=get_noise_std, _batch_mf_err=lambda *a, **kw: 0.0,
        auto_select_gpu=lambda: torch.device('cpu'),
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), '<pipeline-functions>', 'exec'), namespace)
    if not old:
        namespace['select_visible_cuda'] = lambda: torch.device('cpu')
    return namespace


class WaveformKDTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.clean = .03 * torch.randn(1, 2048, device=DEVICE)
        self.noisy = self.clean + .02 * torch.randn_like(self.clean)
        self.sc, self.sn = waveform_batch_to_spec(self.clean, self.noisy, CFG)
        self.teacher_source = (self.clean + .2 * (self.noisy - self.clean)).detach().requires_grad_()
        mag, pha, _ = mag_pha_stft(self.teacher_source, 512, 128, 512, .5)
        self.teacher_wav = mag_pha_istft(mag, pha, 512, 128, 512, .5)

    def test_reconstruction_matches_detached_gt_cache_but_keeps_gradient(self):
        model = student()
        model(self.sc, self.sn)
        wav = student_waveform_with_grad(model, CFG)
        torch.testing.assert_close(wav, model.last_wavs['pred'], rtol=0, atol=0)
        self.assertTrue(wav.requires_grad)
        self.assertFalse(model.last_wavs['pred'].requires_grad)

    def test_time_term_matches_waveform_l1_not_gt_time_scalar(self):
        model = student()
        model(self.sc, self.sn)
        actual = waveform_time_distill_loss(model, self.teacher_wav, CFG)
        expected = F.l1_loss(model.last_wavs['pred'], self.teacher_wav.detach())
        torch.testing.assert_close(actual.detach(), expected, rtol=0, atol=0)
        self.assertTrue(actual.requires_grad)
        self.assertFalse(model.last_losses['time'].requires_grad)

    def test_time_alone_reaches_encoder_mask_and_phase_but_not_teacher(self):
        model = student()
        model(self.sc, self.sn)
        waveform_time_distill_loss(model, self.teacher_wav, CFG).backward()
        for module in (model.dense_encoder, model.mask_decoder, model.phase_decoder):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all().item() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
        self.assertIsNone(self.teacher_source.grad)

    def test_zero_time_weight_preserves_three_term_value_and_all_gradients(self):
        first, second = student(), student()
        first(self.sc, self.sn)
        second(self.sc, self.sn)
        old = response_distill_loss_cross_resolution(first, self.teacher_wav, self.sn, CFG)
        old_total = 3.5 * old['mask'] + 5.0 * old['mag'] + 2.3 * old['com']
        new_total, info = response_kd_with_time(second, self.teacher_wav, self.sn, CFG, dict(WEIGHTS, time=0))
        torch.testing.assert_close(old_total, new_total, atol=0, rtol=0)
        old_total.backward()
        new_total.backward()
        for p, q in zip(first.parameters(), second.parameters()):
            self.assertEqual(p.grad is None, q.grad is None)
            if p.grad is not None:
                # CPU is bit-exact. CUDA convolution reductions can be nondeterministic.
                tolerance = dict(atol=1e-6, rtol=1e-5) if DEVICE.type == 'cuda' else dict(atol=0, rtol=0)
                torch.testing.assert_close(p.grad, q.grad, **tolerance)
        self.assertEqual(info['resp_time'], 0)
        self.assertIsNone(self.teacher_source.grad)

    def test_four_term_weighted_sum_and_diagnostics(self):
        model = student()
        gt = model(self.sc, self.sn)
        kd, info = response_kd_with_time(model, self.teacher_wav, self.sn, CFG, WEIGHTS)
        reference = (3.5 * info['resp_mask'] + 5 * info['resp_mag']
                     + 2.3 * info['resp_com'] + .25 * info['resp_time'])
        # The diagnostic sum uses Python float64; actual KD arithmetic is FP32.
        self.assertTrue(math.isclose(kd.item(), reference, rel_tol=1e-6, abs_tol=1e-7))
        self.assertAlmostEqual(info['resp_time_weighted'], .25 * info['resp_time'])
        total = .48 * gt + .52 * kd
        total.backward()
        self.assertTrue(torch.isfinite(total).item())
        self.assertTrue(all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None))
        self.assertIsNone(self.teacher_source.grad)

    def test_waveform_alignment_does_not_silently_crop_or_pad(self):
        model = student()
        model(self.sc, self.sn)
        for wrong in (self.teacher_wav[:, :-1], self.teacher_wav.unsqueeze(1), self.teacher_wav.repeat(2, 1)):
            with self.subTest(shape=wrong.shape), self.assertRaises(ValueError):
                waveform_time_distill_loss(model, wrong, CFG)

    def test_detached_student_output_is_rejected(self):
        model = student()
        model(self.sc, self.sn)
        model.last_outputs = {k: v.detach() for k, v in model.last_outputs.items()}
        with self.assertRaisesRegex(RuntimeError, 'detached'):
            waveform_time_distill_loss(model, self.teacher_wav, CFG)

    def test_nonfinite_waveform_is_rejected(self):
        model = student()
        model(self.sc, self.sn)
        target = self.teacher_wav.detach().clone()
        target[0, 0] = float('nan')
        with self.assertRaises(FloatingPointError):
            waveform_time_distill_loss(model, target, CFG)

    def test_invalid_time_weight_is_rejected(self):
        model = student()
        model(self.sc, self.sn)
        for value in (-1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                response_kd_with_time(model, self.teacher_wav, self.sn, CFG, dict(WEIGHTS, time=value))

    def test_norm_or_complex_input_mode_cannot_silently_change_units(self):
        model = student()
        model(self.sc, self.sn)
        for key, value in (('norm', '1'), ('fea', 'cpx')):
            changed = copy.deepcopy(CFG)
            changed['model'][key] = value
            with self.assertRaises(ValueError):
                student_waveform_with_grad(model, changed)

    def test_10000_samples_align_to_existing_9984_samples(self):
        model = student()
        clean = self.clean.repeat(1, 5)[:, :10000]
        noisy = self.noisy.repeat(1, 5)[:, :10000]
        sc, sn = waveform_batch_to_spec(clean, noisy, CFG)
        tm, tp, _ = mag_pha_stft(clean, 512, 128, 512, .5)
        teacher = mag_pha_istft(tm, tp, 512, 128, 512, .5)
        model(sc, sn)
        self.assertEqual(tuple(sn.shape), (1, 2, 129, 157))
        self.assertEqual(tuple(student_waveform_with_grad(model, CFG).shape), (1, 9984))
        self.assertEqual(tuple(teacher.shape), (1, 9984))
        self.assertTrue(torch.isfinite(waveform_time_distill_loss(model, teacher, CFG)).item())


class TrainingLoopAndSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='timeKD_synthetic_')
        self.root = Path(self.temp.name)
        self.weights = self.root / 'synthetic_teacher.pth'
        torch.save(FakeTeacher(TCFG).state_dict(), self.weights)
        SyntheticDataset.interrupt_after = None

    def tearDown(self):
        SyntheticDataset.interrupt_after = None
        self.temp.cleanup()

    def train(self, name, time=.25, epochs=2, resume=False, old=False):
        folder = self.root / name
        folder.mkdir(exist_ok=True)
        source = (ROOT / 'pipeline_distill_crossarch_v6_256_64.py').read_text(encoding='utf-8-sig') if old else SOURCE
        namespace = pipeline_namespace(source, old=old)
        with contextlib.redirect_stdout(io.StringIO()):
            namespace['run_crossarch_distillation'](
                copy.deepcopy(TCFG), copy.deepcopy(CFG), str(self.weights), str(self.root / 'synthetic_data'),
                total_epochs=epochs, schedule_epochs=epochs, batch_size=1, accum=2,
                patience=20, seed=42, num_workers=0, cudnn_benchmark=False,
                resp_weights=dict(WEIGHTS, time=time),
                model_save=str(folder / 'best.pth'), log_csv=str(folder / 'log.csv'),
                resume_path=str(folder / 'state.pth'), resume=resume,
            )
        state = torch.load(folder / 'state.pth', map_location='cpu', weights_only=False)
        with (folder / 'log.csv').open(newline='') as f:
            rows = list(csv.reader(f))
        return state, rows

    def test_zero_time_actual_training_loop_matches_original_weights_and_csv(self):
        old, old_csv = self.train('original', time=0, old=True)
        new, new_csv = self.train('zero_time', time=0)
        for key in old['student_state_dict']:
            torch.testing.assert_close(old['student_state_dict'][key], new['student_state_dict'][key], atol=0, rtol=0)
        self.assertEqual(old_csv, [row[:14] for row in new_csv])
        self.assertEqual(new['experiment_metadata']['settings']['resp_weights']['time'], 0)
        self.assertIsNotNone(new['best_epoch'])

    def test_interrupted_run_resumes_same_trajectory_and_csv_as_continuous(self):
        full, full_csv = self.train('continuous', epochs=3)
        SyntheticDataset.interrupt_after = 6
        with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
            self.train('split', epochs=3)
        SyntheticDataset.interrupt_after = None
        resumed, resumed_csv = self.train('split', epochs=3, resume=True)
        for key in full['student_state_dict']:
            torch.testing.assert_close(full['student_state_dict'][key], resumed['student_state_dict'][key], atol=0, rtol=0)
        self.assertEqual(full_csv, resumed_csv)
        self.assertEqual(resumed['epoch'], 3)

    def test_time_logging_and_zero_alpha_skip(self):
        state, rows = self.train('logging', epochs=2)
        header, first, last = rows
        first, last = dict(zip(header, first)), dict(zip(header, last))
        self.assertGreater(float(first['resp_time']), 0)
        self.assertAlmostEqual(float(first['resp_time_weighted']), .25 * float(first['resp_time']))
        self.assertAlmostEqual(float(first['effective_time_kd']), .5 * float(first['resp_time_weighted']))
        self.assertEqual(float(last['alpha']), 0)
        self.assertEqual(float(last['train_kd']), 0)
        self.assertEqual(float(last['effective_time_kd']), 0)
        self.assertEqual(float(last['w_resp_time']), .25)

    def test_resume_rejects_changed_time_weight(self):
        self.train('weight')
        with self.assertRaisesRegex(ValueError, 'metadata mismatch'):
            self.train('weight', time=.5, resume=True)

    def test_fresh_refuses_existing_outputs(self):
        self.train('exists')
        before = (self.root / 'exists/best.pth').read_bytes()
        with self.assertRaises(FileExistsError):
            self.train('exists')
        self.assertEqual(before, (self.root / 'exists/best.pth').read_bytes())

    def test_resume_refuses_old_three_term_state(self):
        self.train('old', time=0, old=True)
        with self.assertRaisesRegex(ValueError, 'metadata mismatch'):
            self.train('old', time=0, resume=True)

    def test_resume_rejects_duplicate_or_extra_epoch_in_csv(self):
        self.train('csv')
        path = self.root / 'csv/log.csv'
        with path.open(newline='') as f:
            rows = list(csv.reader(f))
        with path.open('a', newline='') as f:
            csv.writer(f).writerow(rows[-1])
        with self.assertRaisesRegex(ValueError, 'CSV epochs'):
            self.train('csv', resume=True)

    def test_metadata_rejects_seed_config_teacher_and_schedule_changes(self):
        state, _ = self.train('metadata')
        signature = state['experiment_metadata']
        validate = pipeline_namespace()['validate_resume_state']
        for key in ('settings', 'student_config', 'teacher_weights_sha256', 'source_sha256'):
            changed = copy.deepcopy(signature)
            changed[key] = 'different'
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'metadata mismatch'):
                validate(state, changed, str(self.root / 'metadata/log.csv'))

    def test_guard_requires_distinct_paths_and_complete_resume_outputs(self):
        guard = pipeline_namespace()['guard_output_paths']
        with self.assertRaises(ValueError):
            guard('same', 'same', 'other', False)
        paths = [str(self.root / name) for name in ('best', 'resume', 'csv')]
        with self.assertRaises(FileNotFoundError):
            guard(*paths, True)
        guard(*paths, False)

    def test_atomic_save_failure_preserves_old_checkpoint(self):
        save = pipeline_namespace()['atomic_torch_save']
        target = self.root / 'atomic.pth'
        target.write_bytes(b'previous-complete-file')
        with mock.patch.object(torch, 'save', side_effect=RuntimeError('synthetic write failure')):
            with self.assertRaises(RuntimeError):
                save({'value': 1}, str(target))
        self.assertEqual(target.read_bytes(), b'previous-complete-file')
        self.assertEqual(list(self.root.glob('.timeKD-*.tmp')), [])

    def test_cuda_selector_refuses_fallback_and_preserves_visibility(self):
        node = next(n for n in ast.parse(SOURCE).body if isinstance(n, ast.FunctionDef) and n.name == 'select_visible_cuda')
        ns = dict(torch=torch, os=os)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<cuda-selector>', 'exec'), ns)
        with mock.patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'will not fall back to CPU'):
                ns['select_visible_cuda']()
        with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'allocated-GPU'}):
            with mock.patch.object(torch.cuda, 'is_available', return_value=True), mock.patch.object(torch.cuda, 'set_device'), mock.patch.object(torch.cuda, 'get_device_name', return_value='Mock GPU'), mock.patch.object(torch, 'empty', return_value=torch.zeros(1)):
                self.assertEqual(ns['select_visible_cuda'](), torch.device('cuda:0'))
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'allocated-GPU')

    def test_cli_defaults_and_original_config_are_preserved(self):
        tree = ast.parse(SOURCE)
        flags = {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'add_argument' and node.args and isinstance(node.args[0], ast.Constant)):
                for kw in node.keywords:
                    if kw.arg == 'default':
                        flags[node.args[0].value] = ast.literal_eval(kw.value)
        self.assertEqual(flags['--batch_size'], 4)
        self.assertEqual(flags['--accum'], 4)
        self.assertEqual(flags['--w_resp_time'], .25)
        self.assertIn('timeKD', flags['--model_save'])
        self.assertEqual(CFG['model']['loss_weights'], dict(time=.25, com=.4, con=.25, mr=.08))

    def test_wrong_grid_noncausal_or_sample_rate_is_rejected_before_training(self):
        run = pipeline_namespace()['run_crossarch_distillation']
        for key, value in (('hop_size', 128), ('causal', False), ('sampling_rate', 2000)):
            cfg = copy.deepcopy(CFG)
            cfg['model'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                run(TCFG, cfg, str(self.weights), 'synthetic', resp_weights=WEIGHTS)

    def test_unsupported_weights_and_features_fail_instead_of_being_ignored(self):
        run = pipeline_namespace()['run_crossarch_distillation']
        for kwargs in (dict(w_response=.5), dict(w_feature=1), dict(use_similarity=True),
                       dict(no_kd=True), dict(resp_weights=dict(WEIGHTS, pha=.1))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run(TCFG, CFG, str(self.weights), 'synthetic', **kwargs)


class ShellGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Windows System32/bash.exe is a WSL launcher, not a native Git Bash.
        git_bash = Path('C:/Program Files/Git/bin/bash.exe')
        cls.bash = str(git_bash) if os.name == 'nt' and git_bash.is_file() else shutil.which('bash')
        if not cls.bash:
            raise unittest.SkipTest('Bash not installed')
        cls.script = PACKAGE / 'run_student_causal_KD_256_64_timeKD.sh'

    def prefix(self, root, resume='0', **extra):
        # Windows has no flock: exercise guard logic with an explicit stub,
        # not a claim that real Linux lock contention was tested locally.
        source = self.script.read_text(encoding='utf-8').split('# Compute-node preflight', 1)[0]
        source = 'flock() { [[ "${TEST_LOCK_FAIL:-0}" != 1 ]]; }\n' + source
        env = dict(os.environ, PROJ=root.as_posix(), RESUME=resume, W_TIME='0.25', SEED='42', SMOKE_ONLY='0', SLURM_JOB_ID='synthetic-1', **extra)
        # Binary stdin avoids Windows subprocess translating LF into CRLF.
        checked = subprocess.run([self.bash], input=(source + '\nprintf "GUARD_OK:%s\\n" "${RESUME_ARGS[*]}"\n').encode('utf-8'), capture_output=True, env=env)
        checked.stdout = checked.stdout.decode('utf-8', errors='replace')
        checked.stderr = checked.stderr.decode('utf-8', errors='replace')
        return checked

    def test_embedded_preflight_python_compiles(self):
        source = self.script.read_text(encoding='utf-8')
        preflight = source.split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        compile(preflight, '<shell-preflight>', 'exec')

    @staticmethod
    def paths(root):
        stem = 'student_16ch_1blk_respKD_causal_256_64_teacher_v2_timeKD_w0p25_seed42'
        return [root / 'model_weight' / (stem + '.pth'), root / 'model_weight' / (stem + '.resume_state.pth'), root / 'logs' / ('log_' + stem + '.csv')]

    def test_bash_syntax_and_lf(self):
        for filename in ('run_student_causal_KD_256_64_timeKD.sh', 'run_smoke_timeKD_256_64.sh'):
            path = PACKAGE / filename
            self.assertNotIn(b'\r', path.read_bytes())
            checked = subprocess.run([self.bash, '-n', path.as_posix()], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_direct_login_execution_is_refused(self):
        env = dict(os.environ)
        env.pop('SLURM_JOB_ID', None)
        checked = subprocess.run([self.bash, self.script.as_posix()], capture_output=True, text=True, env=env)
        self.assertNotEqual(checked.returncode, 0)
        self.assertIn('login node', checked.stderr)

    def test_fresh_refuses_each_existing_output(self):
        with tempfile.TemporaryDirectory(prefix='timeKD_shell_') as name:
            self.assertEqual(self.prefix(Path(name)).returncode, 0)
        for index in range(3):
            with self.subTest(index=index), tempfile.TemporaryDirectory(prefix='timeKD_shell_') as name:
                root = Path(name)
                path = self.paths(root)[index]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'synthetic-existing-file')
                checked = self.prefix(root)
                self.assertNotEqual(checked.returncode, 0)
                self.assertIn('refuses to overwrite', checked.stderr)
                self.assertEqual(path.read_bytes(), b'synthetic-existing-file')

    def test_resume_requires_all_three_outputs(self):
        for bits in range(8):
            with self.subTest(bits=bits), tempfile.TemporaryDirectory(prefix='timeKD_shell_') as name:
                root = Path(name)
                for index, path in enumerate(self.paths(root)):
                    if bits & (1 << index):
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b'nonempty-synthetic-fixture')
                checked = self.prefix(root, resume='1')
                self.assertEqual(checked.returncode == 0, bits == 7, checked.stderr)
                if bits == 7:
                    self.assertIn('--resume --resume_path', checked.stdout)

    def test_invalid_resume_and_unavailable_lock_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='timeKD_shell_') as name:
            self.assertNotEqual(self.prefix(Path(name), resume='wrong').returncode, 0)
            checked = self.prefix(Path(name), TEST_LOCK_FAIL='1')
            self.assertNotEqual(checked.returncode, 0)
            self.assertIn('another job', checked.stderr)


@unittest.skipUnless(ARGS.teacher_smoke, 'Real Teacher/CUDA smoke not requested; local CPU tests are not GPU validation')
class RealTeacherSmoke(unittest.TestCase):
    def test_actual_teacher_checkpoint_four_term_backward(self):
        sys.path.insert(0, str(ROOT / 'MECG-E'))
        from models.SSEMGNet import SSEMGNet
        path = ARGS.teacher_weights or ROOT / 'model_weight/v19_tt_mask_v2_mix_nv0_weights.pth'
        model_t = SSEMGNet(copy.deepcopy(TCFG)).to(DEVICE).eval()
        model_t.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False), strict=True)
        for parameter in model_t.parameters():
            parameter.requires_grad_(False)
        torch.manual_seed(42)
        clean = .03 * torch.randn(1, 10000, device=DEVICE)
        noisy = clean + .02 * torch.randn_like(clean)
        tc, tn = waveform_batch_to_spec(clean, noisy, TCFG)
        sc, sn = waveform_batch_to_spec(clean, noisy, CFG)
        with torch.no_grad():
            model_t(tc, tn)
        model_s = student().train()
        gt = model_s(sc, sn)
        kd, info = response_kd_with_time(model_s, model_t.last_wavs['pred'], sn, CFG, WEIGHTS)
        total = .48 * gt + .52 * kd
        total.backward()
        self.assertTrue(torch.isfinite(total).item())
        self.assertGreater(info['resp_time'], 0)
        self.assertTrue(all(torch.isfinite(p.grad).all().item() for p in model_s.parameters() if p.grad is not None))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model_s.parameters()))
        self.assertTrue(all(p.grad is None for p in model_t.parameters()))
        print(f'[REAL TEACHER CUDA SMOKE PASS] time={info["resp_time"]:.8g} weighted={info["resp_time_weighted"]:.8g}', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    print(f'[test] torch={torch.__version__} device={DEVICE} real_teacher={ARGS.teacher_smoke}', flush=True)
    unittest.main(argv=[__file__] + TEST_ARGS, verbosity=2)
