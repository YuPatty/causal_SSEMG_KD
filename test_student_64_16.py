"""Local synthetic regression tests for the new grid and shared-support KD.

Run on a workstation, or submit on a compute node; no experiment files written.
Does not instantiate the real Mamba Teacher or claim test-set performance.
"""
from pathlib import Path
import argparse
import copy
import importlib.util
import sys
import unittest
from types import SimpleNamespace

p=argparse.ArgumentParser()
p.add_argument('--project-root',type=Path,required=True)
args,rest=p.parse_known_args()
root=args.project_root.resolve()
sys.path.insert(0,str(root))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
import torch.nn.functional as F
import yaml
from cross_resolution_64_16 import waveform_batch_to_spec,scoring_prefix_length
from distill_loss_v3_64_16 import _student_grid_targets,response_distill_loss_cross_resolution
from spectrogram_utils import mag_pha_stft,mag_pha_istft

torch.set_num_threads(1)
cfg=yaml.safe_load((root/'config/student_16ch_1blk_causal_64_16.yaml').read_text(encoding='utf-8-sig'))
ref=yaml.safe_load((root/'config/student_16ch_1blk_causal_128_32.yaml').read_text(encoding='utf-8-sig'))
spec=importlib.util.spec_from_file_location('student6416_regression',root/'MECG-E/models/StudentNet.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class GridTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.clean=.03*torch.randn(1,10000)
        self.noisy=self.clean+.02*torch.sin(2*torch.pi*30*torch.arange(10000)/1000)[None,:]
        self.cs,self.ns=waveform_batch_to_spec(self.clean,self.noisy,cfg)

    def test_only_model_stft_changes(self):
        expected=copy.deepcopy(ref)
        expected['model'].update(n_fft=64,hop_size=16,win_size=64)
        self.assertEqual(cfg,expected)
        self.assertEqual(cfg['metrics'],dict(sampling_rate=1000,n_fft=256,hop_size=32,win_size=128,segment_sec=1))

    def test_native_reconstruction_and_teacher_support(self):
        self.assertEqual(tuple(self.ns.shape),(1,2,33,626))
        native=mag_pha_istft(self.ns[:,0],self.ns[:,1],64,16,64,.5)
        self.assertEqual(tuple(native.shape),(1,10000))
        torch.testing.assert_close(native,self.noisy,atol=1e-7,rtol=1e-5)
        mag,pha,_=mag_pha_stft(self.noisy,512,128,512,.5)
        teacher=mag_pha_istft(mag,pha,512,128,512,.5)
        self.assertEqual(tuple(teacher.shape),(1,9984))
        targets=_student_grid_targets(teacher,self.ns,cfg)
        self.assertEqual(tuple(targets[0].shape),(1,33,625))

    def test_response_values_equal_manual_common_frame_oracle(self):
        teacher=self.clean[:,:9984].clone().requires_grad_()
        # Leaf tensors let us check exactly which response samples receive KD.
        mask=torch.rand(1,1,626,33,requires_grad=True)
        mag=torch.rand(1,33,626,requires_grad=True)
        com=torch.rand(1,33,626,2,requires_grad=True)
        gate=torch.zeros(1,1,1,33);gate[...,:9]=1
        student=SimpleNamespace(last_outputs=dict(mask=mask,mag_g_FT=mag,com_g=com),
                                mask_decoder=SimpleNamespace(low_gate=gate))
        losses=response_distill_loss_cross_resolution(student,teacher,self.ns,cfg)
        tm,tp,_=mag_pha_stft(teacher.detach(),64,16,64,.5)
        target_mask=(tm/self.ns[:,0,:,:625].clamp_min(1e-8)).clamp(0,1)
        target_mask=target_mask.transpose(1,2).unsqueeze(1)*gate+(1-gate)
        tc=torch.stack((tm*torch.cos(tp),tm*torch.sin(tp)),dim=-1)
        torch.testing.assert_close(losses['mask'],F.mse_loss(mask[:,:,:625],target_mask))
        torch.testing.assert_close(losses['mag'],F.mse_loss(mag[:,:,:625],tm))
        torch.testing.assert_close(losses['com'],F.mse_loss(com[:,:,:625],tc))
        (3.5*losses['mask']+5*losses['mag']+2.3*losses['com']).backward()
        self.assertIsNone(teacher.grad)
        for value in (mask,mag,com):self.assertTrue(torch.isfinite(value.grad).all())
        self.assertEqual(float(mask.grad[:,:,625:].abs().sum()),0)
        self.assertEqual(float(mag.grad[:,:,625:].abs().sum()),0)
        self.assertEqual(float(com.grad[:,:,625:].abs().sum()),0)
        self.assertGreater(float(mag.grad[:,:,:625].abs().sum()),0)
        self.assertEqual(tuple(student.last_outputs['mask'].shape),(1,1,626,33))

    def test_full_student_gt_plus_shared_support_kd_backward(self):
        model=module.StudentSSEMGNet(cfg).train()
        gt=model(self.cs,self.ns)
        self.assertEqual(tuple(model.last_wavs['pred'].shape),(1,10000))
        self.assertEqual(int(model.mask_decoder.low_gate.sum()),9)
        losses=response_distill_loss_cross_resolution(model,self.clean[:,:9984],self.ns,cfg)
        total=.5*gt+.5*(3.5*losses['mask']+5*losses['mag']+2.3*losses['com'])
        self.assertTrue(torch.isfinite(total))
        total.backward()
        gradients=[p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(float(model.dense_encoder.dense_conv_1[0].weight.grad.abs().sum()),0)

    def test_time_mismatch_outside_teacher_reconstruction_tail_is_rejected(self):
        with self.assertRaises(RuntimeError):_student_grid_targets(self.clean[:,:9728],self.ns,cfg)
        with self.assertRaises(ValueError):_student_grid_targets(self.clean[:,:9985],self.ns,cfg)
        with self.assertRaises(RuntimeError):_student_grid_targets(torch.zeros(1,10112),self.ns,cfg)

    def test_identical_support_requires_no_padding_or_interpolation(self):
        same=waveform_batch_to_spec(self.clean[:,:9984],self.noisy[:,:9984],cfg)[1]
        target=_student_grid_targets(self.clean[:,:9984],same,cfg)
        self.assertEqual(target[0].shape,same[:,0].shape)

    def test_shared_evaluation_duration(self):
        self.assertEqual(scoring_prefix_length(10000),9984)
        self.assertEqual(scoring_prefix_length(9984),9984)
        self.assertEqual(scoring_prefix_length(10112),10112)
        with self.assertRaises(ValueError):scoring_prefix_length(127)


if __name__=='__main__':unittest.main(argv=[sys.argv[0],*rest],verbosity=2)
