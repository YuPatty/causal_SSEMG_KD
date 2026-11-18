"""Read-only config/source checks and disposable 64/16 forward/backward smoke.

No training checkpoint or CSV is written. Uses the existing Student and STFT
sources from --project-root. Actual training remains the original noKD branch.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--student-config',type=Path)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--mode',choices=('train','smoke','infer'),default='smoke')
    p.add_argument('--weights',type=Path)
    p.add_argument('--resume-state',type=Path)
    p.add_argument('--allow-local-cpu',action='store_true',help='Local workstation QA only, never on an HPC login node')
    return p.parse_args()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    import torch
    import yaml
    root=args.project_root.resolve()
    cfg_path=(args.student_config or root/'config/student_16ch_1blk_causal_64_16.yaml').resolve()
    required=['pipeline_distill_crossarch_v6_64_16.py','distill_loss_v3_64_16.py',
              'cross_resolution_64_16.py','spectrogram_utils.py','spectrogram_dataset.py',
              'pipeline_spectrogram.py','MECG-E/models/StudentNet.py',
              'config/config_spectrogram_v19_tt_mask.yaml',
              'config/student_16ch_1blk_causal_256_64.yaml']
    if args.mode=='infer':required.append('inference_student_patched_v6_64_16.py')
    paths=[root/name for name in required]+[cfg_path]
    for path in paths:
        if not path.is_file():raise FileNotFoundError(path)
    before={str(p):digest(p) for p in paths}
    base=yaml.safe_load((root/'config/student_16ch_1blk_causal_256_64.yaml').read_text(encoding='utf-8-sig'))
    cfg=yaml.safe_load(cfg_path.read_text(encoding='utf-8-sig'))
    if tuple(base['model'][k] for k in ('n_fft','hop_size','win_size'))!=(256,64,256):
        raise ValueError('Reference must be the original 256/64/256 config')
    if base['model'].get('causal') is not True:raise ValueError('Reference model must be causal')
    expected=copy.deepcopy(base)
    expected['model'].update(n_fft=64,hop_size=16,win_size=64)
    if cfg!=expected:
        raise ValueError('64/16 config must differ ONLY in model.n_fft, model.hop_size, model.win_size; metrics and loss settings must match')
    if args.data_root:
        splits=('test',) if args.mode=='infer' else ('train','valid')
        for split in splits:
            for kind in ('clean','noisy'):
                p=args.data_root/split/kind
                if not p.is_dir():raise FileNotFoundError(p)
    if args.allow_local_cpu:
        device=torch.device('cpu')
    else:
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; refusing CPU training/inference fallback')
        device=torch.device('cuda:0')
        torch.empty(1,device=device)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    if device.type=='cuda':torch.cuda.manual_seed_all(42)
    # Only this disposable check process. The unchanged training entry point
    # seeds again and keeps its original cuDNN/optimizer settings.
    torch.backends.cudnn.benchmark=False
    sys.path.insert(0,str(root))
    path=root/'MECG-E/models/StudentNet.py'
    spec=importlib.util.spec_from_file_location('student_6416_preflight_model',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.StudentSSEMGNet(cfg).to(device)
    if getattr(model,'causal',False) is not True:raise ValueError('Loaded Student implementation is not causal')
    if args.weights:
        state=torch.load(args.weights,map_location=device,weights_only=False)
        if isinstance(state,dict) and isinstance(state.get('state_dict'),dict):state=state['state_dict']
        model.load_state_dict(state,strict=True)
    if args.resume_state:
        state=torch.load(args.resume_state,map_location='cpu',weights_only=False)
        model.load_state_dict(state['student_state_dict'],strict=True)
        print(f"[resume check] epoch={state['epoch']} best_val={state['best_val']} no_improve={state['no_improve']}",flush=True)
    from cross_resolution_64_16 import waveform_batch_to_spec
    batch=4 if args.mode!='infer' else 1
    accum=4 if args.mode!='infer' else 1
    optim=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.8,.99),weight_decay=1e-3)
    model.train(args.mode!='infer')
    for micro in range(accum):
        clean=.03*torch.randn(batch,10000,device=device)
        t=torch.arange(10000,device=device)/1000
        noisy=clean+.02*torch.sin(2*torch.pi*30*t)[None,:]
        clean_spec,noisy_spec=waveform_batch_to_spec(clean,noisy,cfg)
        if tuple(noisy_spec.shape)!=(batch,2,33,626):raise ValueError(f'Unexpected spectrogram shape: {tuple(noisy_spec.shape)}')
        with torch.set_grad_enabled(args.mode!='infer'):
            gt=model(clean_spec,noisy_spec)
            if not torch.isfinite(gt):raise ValueError('Non-finite GT loss')
            if args.mode!='infer':(gt/accum).backward()
        wav=model.last_wavs['pred']
        if tuple(wav.shape)!=(batch,10000) or not torch.isfinite(wav).all():
            raise ValueError(f'Unexpected waveform: {tuple(wav.shape)}')
        gate=model.mask_decoder.low_gate
        if tuple(gate.shape)!=(1,1,1,33) or int(gate.sum())!=9:raise ValueError('Unexpected mask gate on the new grid')
    if args.mode!='infer':
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):raise ValueError('Invalid GT gradients')
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        if not torch.isfinite(grad_norm):raise ValueError('Non-finite gradient norm')
        optim.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Optimizer produced non-finite weights')
    for p,value in before.items():
        if digest(Path(p))!=value:raise RuntimeError(f'Source changed during check: {p}')
    print(f"[preflight] torch={torch.__version__} device={device}; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','(unset)')}",flush=True)
    if device.type=='cuda':print('[GPU] '+torch.cuda.get_device_name(0),flush=True)
    print('[config] Only model STFT changed: 256/64/256 -> 64/16/64; common metrics and GT weights unchanged',flush=True)
    print(f'[smoke] batch={batch}, accum={accum}, input={tuple(noisy_spec.shape)}, output={tuple(wav.shape)}, mask_bins=9',flush=True)
    print('[GT components] '+json.dumps({k:float(v) for k,v in model.last_losses.items()}),flush=True)
    for p,value in before.items():print(f'[sha256] {value}  {p}',flush=True)
    print('[64/16 CHECK PASS] Disposable synthetic check only; no checkpoint/CSV writes and no performance claim',flush=True)


if __name__=='__main__':
    args=arguments()
    if not os.environ.get('SLURM_JOB_ID') and not args.allow_local_cpu:
        raise SystemExit('Use sbatch on the cluster; no model work on a login node')
    if args.allow_local_cpu and (args.mode!='smoke' or os.environ.get('SLURM_JOB_ID')):
        raise SystemExit('--allow-local-cpu is only for local smoke QA')
    main(args)
