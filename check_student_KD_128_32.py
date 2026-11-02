"""Allocated-GPU smoke for the actual frozen Teacher ->128/32 response KD.

Uses real Teacher checkpoint and existing models. No checkpoint/CSV writes.
"""
import argparse
from pathlib import Path
import os
import sys
from types import SimpleNamespace


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--student-config',type=Path)
    p.add_argument('--teacher-weights',type=Path,required=True)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--mode',choices=('train','smoke'),default='smoke')
    p.add_argument('--resume-state',type=Path)
    return p.parse_args()


def main(args):
    import torch
    import yaml
    import importlib.util
    import check_student_noKD_128_32 as base
    root=args.project_root.resolve()
    teacher_weights=args.teacher_weights.resolve()
    expected='7b1f97a88449e2d9312d679846ea11f9108622e51e3d1be326463bcf2d139d04'
    if base.digest(teacher_weights)!=expected:
        raise ValueError('Teacher checkpoint SHA256 differs from the verified v2 best-val checkpoint')
    here=Path(__file__).resolve().parent
    paths=[here/name for name in ('pipeline_distill_crossarch_v6_128_32.py','distill_loss_v3_128_32.py','cross_resolution_128_32.py')]
    paths+=[root/'MECG-E/models/SSEMGNet.py',teacher_weights]
    before={str(p):base.digest(p) for p in paths}
    # Validate the shared candidate config, real GT gradients and CUDA first.
    base.main(SimpleNamespace(project_root=root,student_config=args.student_config,
        data_root=args.data_root,mode='smoke',weights=None,
        resume_state=args.resume_state,allow_local_cpu=False))
    sys.path.insert(0,str(root))
    sys.path.insert(0,str(root/'MECG-E'))
    from models.SSEMGNet import SSEMGNet
    from models.StudentNet import StudentSSEMGNet
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    cross=load('preflight_12832_cross',here/'cross_resolution_128_32.py')
    lossmod=load('preflight_12832_loss',here/'distill_loss_v3_128_32.py')
    teacher_cfg=yaml.safe_load((root/'config/config_spectrogram_v19_tt_mask.yaml').read_text(encoding='utf-8-sig'))
    cfg_path=(args.student_config or root/'config/student_16ch_1blk_causal_128_32.yaml').resolve()
    student_cfg=yaml.safe_load(cfg_path.read_text(encoding='utf-8-sig'))
    if tuple(teacher_cfg['model'][k] for k in ('n_fft','hop_size','win_size'))!=(512,128,512):
        raise ValueError('Expected512/128/512 Teacher')
    device=torch.device('cuda:0')
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    teacher=SSEMGNet(teacher_cfg).to(device).eval()
    teacher.load_state_dict(torch.load(teacher_weights,map_location=device,weights_only=False),strict=True)
    teacher.requires_grad_(False)
    frozen={k:v.detach().cpu().clone() for k,v in teacher.state_dict().items()}
    student=StudentSSEMGNet(student_cfg).to(device).train()
    if args.resume_state:
        state=torch.load(args.resume_state,map_location=device,weights_only=False)
        student.load_state_dict(state['student_state_dict'],strict=True)
    optim=torch.optim.AdamW(student.parameters(),lr=3e-4,betas=(.8,.99),weight_decay=1e-3)
    alpha=float(lossmod.annealed_alpha(1,150,schedule='linear'))
    print('[KD stage] real frozen Teacher checkpoint; Student128/32; batch4 accum4',flush=True)
    for _ in range(4):
        clean=.03*torch.randn(4,10000,device=device)
        t=torch.arange(10000,device=device)/1000
        noisy=clean+.02*torch.sin(2*torch.pi*30*t)[None,:]
        sc,sn=cross.waveform_batch_to_spec(clean,noisy,student_cfg)
        tc,tn=cross.waveform_batch_to_spec(clean,noisy,teacher_cfg)
        gt=student(sc,sn)
        with torch.no_grad():teacher(tc,tn)
        teacher_wav=teacher.last_wavs['pred']
        if tuple(teacher_wav.shape)!=(4,9984):raise ValueError('Unexpected Teacher waveform shape')
        losses=lossmod.response_distill_loss_cross_resolution(student,teacher_wav,sn,student_cfg)
        kd=3.5*losses['mask']+5.*losses['mag']+2.3*losses['com']
        total=(1-alpha)*gt+alpha*kd
        if not torch.isfinite(total) or not all(torch.isfinite(v) for v in losses.values()):
            raise ValueError('Non-finite response KD loss')
        (total/4).backward()
    grads=[p.grad for p in student.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):raise ValueError('Invalid Student KD gradients')
    if any(p.grad is not None for p in teacher.parameters()):raise ValueError('Teacher received gradients')
    norm=torch.nn.utils.clip_grad_norm_(student.parameters(),5.)
    if not torch.isfinite(norm):raise ValueError('Invalid KD gradient norm')
    optim.step()
    if not all(torch.isfinite(p).all() for p in student.parameters()):raise ValueError('Invalid disposable optimizer step')
    for name,value in teacher.state_dict().items():
        if not torch.equal(value.detach().cpu(),frozen[name]):raise ValueError('Teacher state changed')
    for p,value in before.items():
        if base.digest(Path(p))!=value:raise ValueError('Source/checkpoint changed during smoke')
    print(f"[KD smoke] alpha={alpha:.8f}, mask={float(losses['mask']):.8g}, mag={float(losses['mag']):.8g}, complex={float(losses['com']):.8g}",flush=True)
    print('[KD teacher sha256] '+expected,flush=True)
    print('[128/32 KD CHECK PASS] Actual Teacher load/forward, re-analysis and Student backward passed; no checkpoint/CSV writes; no performance claim',flush=True)


if __name__=='__main__':
    args=arguments()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Use sbatch on the cluster; this actual-Teacher check requires an allocated GPU')
    main(args)
