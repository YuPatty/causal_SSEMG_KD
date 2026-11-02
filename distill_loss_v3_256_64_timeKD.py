"""Original mask/magnitude/complex response KD plus waveform L1 KD.

No StudentNet edits are needed. Reconstruct Student output from last_outputs,
NOT last_wavs['pred'] (which is intentionally detached for metric logging).
Teacher uses its own 512/128 reconstruction; no resampling, shift correction,
independent amplitude normalization, padding or silent truncation is applied.
"""
import math

import torch
import torch.nn.functional as F

from distill_loss_v3_256_64 import response_distill_loss_cross_resolution
from spectrogram_utils import mag_pha_istft


def student_waveform_with_grad(student, student_cfg):
    """Use the same FP32, decompressed iSTFT as the Student GT time loss."""
    h = student_cfg['model']
    if h.get('fea', 'pha') != 'pha' or h.get('norm', False) is not False:
        raise ValueError('time KD currently requires fea=pha and norm=False')
    out = student.last_outputs
    mag, pha = out['mag_g_FT'], out['pha_g']
    if mag.ndim != 3 or mag.shape != pha.shape:
        raise ValueError('Expected matching Student magnitude/phase [B,F,T]')
    # .float() preserves the graph. Do not detach or wrap this in no_grad().
    with torch.autocast(device_type=mag.device.type, enabled=False):
        return mag_pha_istft(
            mag.float(), pha.float(), n_fft=int(h['n_fft']),
            hop_size=int(h['hop_size']), win_size=int(h['win_size']),
            compress_factor=float(h.get('compress_factor', 1.0)),
        )


def waveform_time_distill_loss(student, teacher_wav, student_cfg):
    """Mean absolute Student--stopgrad(Teacher) error in waveform units."""
    pred = student_waveform_with_grad(student, student_cfg)
    target = teacher_wav.detach().float()
    if pred.ndim != 2 or target.ndim != 2 or pred.shape != target.shape:
        raise ValueError(
            f'time KD needs aligned [B,N] waveforms, got Student={tuple(pred.shape)} '
            f'Teacher={tuple(target.shape)}; do not silently crop/pad/shift them'
        )
    if pred.device != target.device:
        raise ValueError('Student and Teacher waveforms must use the same device')
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise FloatingPointError('Non-finite waveform in time KD')
    if torch.is_grad_enabled() and not pred.requires_grad:
        raise RuntimeError('Student waveform is detached: time KD would not train it')
    return F.l1_loss(pred, target, reduction='mean')


def response_kd_with_time(student, teacher_wav, student_noisy_spec,
                          student_cfg, resp_weights):
    """Return weighted four-term KD and detached scalar diagnostics.

    w_time=0 skips reconstruction and preserves the original three-term value
    AND gradient exactly. Independent phase/feature KD remain disabled.
    """
    weights = {k: float(resp_weights.get(k, 0.0))
               for k in ('mask', 'mag', 'com', 'time')}
    if any(not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError('Response KD weights must be finite and nonnegative')
    losses = response_distill_loss_cross_resolution(
        student, teacher_wav, student_noisy_spec, student_cfg,
    )
    kd = (weights['mask'] * losses['mask'] + weights['mag'] * losses['mag']
          + weights['com'] * losses['com'])
    loss_time = torch.zeros((), dtype=kd.dtype, device=kd.device)
    if weights['time'] > 0:
        loss_time = waveform_time_distill_loss(student, teacher_wav, student_cfg)
        kd = kd + weights['time'] * loss_time
    info = {
        'kd_loss': float(kd.detach()),
        'resp_mask': float(losses['mask'].detach()),
        'resp_mag': float(losses['mag'].detach()),
        'resp_pha': 0.0,
        'resp_com': float(losses['com'].detach()),
        'resp_time': float(loss_time.detach()),
        'resp_time_weighted': weights['time'] * float(loss_time.detach()),
        'feat_loss': 0.0, 'rel_loss': None, 'sim_loss': 0.0,
    }
    return kd, info
