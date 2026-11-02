"""FCN/MSEMG/SDEMG native-waveform benchmark, matching the Student/Teacher timer.

Snapshots originate from the user's previous paper/benchmark package. The old
benchmark script is never imported: its global patches/timer remain untouched.
MSEMG class bodies are unchanged; only old import/API bindings are adapted in
this process to the same installed Mamba runtime used by the Teacher benchmark.
"""
from __future__ import annotations
import argparse
import ast
from collections import namedtuple
import functools
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",type=Path,required=True)
    p.add_argument("--models",nargs="+",choices=("fcn","msemg","sdemg64","sdemg128"),
                   default=["fcn","msemg","sdemg128"])
    p.add_argument("--device",choices=("both","cpu","cuda"),default="both")
    p.add_argument("--fcn-weights",type=Path)
    p.add_argument("--msemg-weights",type=Path)
    p.add_argument("--sdemg64-weights",type=Path)
    p.add_argument("--sdemg128-weights",type=Path)
    p.add_argument("--student-helper",type=Path)
    p.add_argument("--teacher-helper",type=Path)
    p.add_argument("--seconds",type=float,default=10.)
    p.add_argument("--cpu-threads",type=int,default=1)
    p.add_argument("--warmup",type=int,default=5)
    p.add_argument("--cpu-runs",type=int,default=30)
    p.add_argument("--gpu-runs",type=int,default=50)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--allow-local-cpu",action="store_true",help="Local workstation only; never use this on an HPC login node")
    return p.parse_args()


if __name__ == "__main__":
    CLI=arguments()
    if not os.environ.get("SLURM_JOB_ID") and not (CLI.allow_local_cpu and CLI.device=="cpu"):
        raise SystemExit("Use sbatch on the cluster; local CPU requires --device cpu --allow-local-cpu")

import torch
from torch import nn


def load_msemg_classes(path, mamba_class, block_class, rmsnorm_class, init_weights):
    """Execute original class bodies ONLY; bind imports without global patches."""
    tree=ast.parse(path.read_text(encoding="utf-8-sig"),filename=str(path))
    classes=[node for node in tree.body if isinstance(node,ast.ClassDef)]
    if {n.name for n in classes}!={"Conv1d","MambaBlock","HNFBlock","EMGMAMBA"}:
        raise ValueError("Unexpected MSEMG snapshot classes; inspect this source version first")
    block=functools.partial(block_class,mlp_cls=nn.Identity) if "mlp_cls" in inspect.signature(block_class.__init__).parameters else block_class
    namespace=dict(__name__="benchmark_original_msemg_classes",torch=torch,nn=nn,F=torch.nn.functional,
        partial=functools.partial,Mamba=mamba_class,Block=block,RMSNorm=rmsnorm_class,_init_weights=init_weights)
    # Source imports torchsummary only for its __main__ demo. Neither is run.
    exec(compile(ast.Module(body=classes,type_ignores=[]),str(path),"exec"),namespace)
    return namespace["EMGMAMBA"]


class MSEMGWaveform(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model=model
    def forward(self,waveform):
        return self.model(waveform.unsqueeze(1)).squeeze(1)


def load_sdemg_classes(folder):
    """Keep original model/sampler definitions, avoid unrelated import-time deps."""
    import numpy as np
    scope=dict(__name__="benchmark_original_sdemg",torch=torch,nn=nn,
               F=torch.nn.functional,ln=math.log,math=math,np=np,
               partial=functools.partial,namedtuple=namedtuple,
               autocast=torch.cuda.amp.autocast,
               tqdm=lambda iterable,**kwargs:iterable)
    def definitions(name,select):
        path=folder/name
        tree=ast.parse(path.read_text(encoding="utf-8-sig"),filename=str(path))
        nodes=[node for node in tree.body if select(node)]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),scope)
    definitions("utils.py",lambda n:isinstance(n,ast.FunctionDef) and n.name in {"exists","default","identity"})
    definitions("deep_filter_model.py",lambda n:isinstance(n,ast.ClassDef))
    conditional=scope["ConditionalModel"]
    def sampler_node(n):
        return isinstance(n,(ast.FunctionDef,ast.ClassDef)) or (
            isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="ModelPrediction" for t in n.targets))
    definitions("ddpm_1d.py",sampler_node)
    return conditional,scope["GaussianDiffusion1D"]


class SDEMGWaveform(nn.Module):
    """Time the original complete reverse-diffusion sampler, including RNG."""
    def __init__(self,diffusion,steps):
        super().__init__()
        self.diffusion=diffusion
        self.steps=steps
    def forward(self,waveform):
        return self.diffusion.denoise(waveform.unsqueeze(1),denoise_timesteps=self.steps).squeeze(1)


def count_sdemg_ops(graph,waveform,common):
    """Count actual repeated Conv/Linear calls along the full sampler path."""
    calls=[]
    denoiser=graph.diffusion.model
    handle=denoiser.register_forward_pre_hook(lambda module,inputs:calls.append(None))
    try:
        full=common.count_main_ops(graph,waveform)
    finally:
        handle.remove()
    if len(calls)!=graph.steps:
        raise ValueError(f"SDEMG expected {graph.steps} denoiser calls, observed {len(calls)}")
    # Repeated layer calls are kept in the JSON. Conv/Linear shapes are constant.
    used_names={r["name"] for r in full["layers"]}
    active_params=sum(p.numel() for name,m in graph.named_modules() if name in used_names
                      for p in m.parameters(recurse=False))
    return dict(conv_linear_gflops=2*full["conv_linear_macs"]/1e9,
                single_step_conv_linear_gflops=2*full["conv_linear_macs"]/graph.steps/1e9,
                selective_scan_gflops_estimate=0.,denoiser_calls=len(calls),
                executed_conv_linear_params=active_params,layer_counts=full["layers"])


def main(args):
    import importlib.util
    root=args.project_root.resolve()
    here=Path(__file__).resolve().parent
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    if (not math.isfinite(args.seconds) or args.seconds<=0 or args.cpu_threads<1
            or args.warmup<1 or min(args.cpu_runs,args.gpu_runs)<2):
        raise ValueError("Invalid timing settings")
    if args.device in ("both","cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    def module_from(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    common_path=(args.student_helper or root/"benchmark_liteemg_causal.py").resolve()
    common=module_from("waveform_benchmark_common",common_path)
    teacher_helper=(args.teacher_helper or root/"benchmark_teacher_v2.py").resolve()
    costs=module_from("waveform_benchmark_costs",teacher_helper) if "msemg" in args.models else None
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    source_files=[Path(__file__).resolve(),common_path]
    specs={}
    if "fcn" in args.models:
        source=here/"benchmark_fcn_model.py"
        specs["fcn"]=(source,(args.fcn_weights or root/"model_weight/fcn_baseline.pth").resolve())
    if "msemg" in args.models:
        source=here/"benchmark_msemg_model_source.py"
        specs["msemg"]=(source,(args.msemg_weights or root/"model_weight/msemg_baseline.pth").resolve())
        source_files.extend([teacher_helper,root/"MECG-E/models/SSEMGNet.py"])
    sdemg_folder=here/"benchmark_sdemg_sources"
    sdemg_cfg=None
    sdemg_classes=None
    if any(label.startswith("sdemg") for label in args.models):
        import yaml
        sdemg_cfg=yaml.safe_load((sdemg_folder/"default.yaml").read_text(encoding="utf-8-sig"))
        if not sdemg_cfg.get("condition") or sdemg_cfg.get("ddim"):
            raise ValueError("This benchmark expects the original conditional DDPM denoise() config")
        source_files.extend(sdemg_folder/name for name in ("ddpm_1d.py","utils.py","default.yaml"))
        sdemg_classes=load_sdemg_classes(sdemg_folder)
        for label,default_file in (("sdemg64","sdemg_baseline_v2.pth"),("sdemg128","sdemg_baseline_feats128.pth")):
            if label in args.models:
                custom=getattr(args,label+"_weights")
                specs[label]=(sdemg_folder/"deep_filter_model.py",(custom or root/"model_weight"/default_file).resolve())
    for source,weight in specs.values():
        if not source.is_file() or not weight.is_file():
            raise FileNotFoundError(f"Missing model snapshot or checkpoint: {source}; {weight}")
        source_files.append(source)
    hashes={str(p):common.file_hash(p) for p in source_files}
    result=dict(environment=dict(torch=str(torch.__version__),python=platform.python_version(),
        cpu=common.cpu_name(),cpu_threads=torch.get_num_threads(),host=platform.node(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,job_id=os.environ.get("SLURM_JOB_ID")),
        protocol=dict(seconds=args.seconds,sampling_rate=1000,batch_size=1,dtype="float32",seed=args.seed,
        warmup=args.warmup,cpu_runs=args.cpu_runs,gpu_runs=args.gpu_runs,
        scope="native waveform -> enhanced waveform; no STFT/iSTFT added to time-domain models",
        timer="same measure() helper as Student/Teacher; wall clock + GPU synchronization",
        excludes="disk IO, model loading, host/device transfers, GT losses, metrics",
        flops="Conv/Linear MAC*2; Mamba internals included analytically; scan estimate separate",
        streaming_latency="NOT measured; offline clip processing time",
        msemg_import_adapter="original class bodies; runtime Mamba/Block/RMSNorm bound explicitly, no vendored-package sys.path or old benchmark global patches"),
        source_sha256=hashes,cases=[],status="running")
    print("[environment] "+json.dumps(result["environment"]),flush=True)
    print("[protocol] "+json.dumps(result["protocol"]),flush=True)
    n=round(1000*args.seconds)
    if n<512:
        raise ValueError("Use at least 512 input samples")
    wave=.03*torch.randn(1,n,generator=torch.Generator().manual_seed(args.seed))
    devices=[torch.device(d) for d in (["cuda:0","cpu"] if args.device=="both" else ["cuda:0" if args.device=="cuda" else "cpu"])]
    teacher_module=None
    if "msemg" in args.models:
        sys.path.insert(0,str(root))
        teacher_module=common.load_module("waveform_benchmark_device_modes",root/"MECG-E/models/SSEMGNet.py")
        if not callable(getattr(teacher_module,"set_model_device_mode",None)):
            raise RuntimeError("Existing Teacher CPU/GPU compatibility helper is missing; inspect source version, do not overwrite the model")
        from mamba_ssm.modules import mamba_simple
        result["environment"]["mamba_module_file"]=str(Path(mamba_simple.__file__).resolve())
    checkpoint_hashes={}
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x",encoding="utf-8") as f:
            json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    try:
        for label in dict.fromkeys(args.models):
            source,weight=specs[label]
            if label=="fcn":
                model=common.load_module("waveform_benchmark_fcn",source).FCN_01().eval()
                expected_params=137801
            elif label=="msemg":
                EMGMAMBA=load_msemg_classes(source,teacher_module.Mamba,teacher_module.Block,
                                            teacher_module._TritonRMSNorm,teacher_module._init_weights)
                model=EMGMAMBA(in_channels=64,feats=64,n_layer=1).eval()
                expected_params=279937
            else:
                feats=64 if label=="sdemg64" else 128
                model=sdemg_classes[0](feats=feats).eval()
                expected_params=1233857 if feats==64 else 4925313
            digest,_=common.load_weights(model,weight)
            checkpoint_hashes[str(weight)]=digest
            model.requires_grad_(False)
            original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            count=sum(p.numel() for p in model.parameters())
            if count!=expected_params:
                raise ValueError(f"{label}: parameter count differs from the previous paper setup: {count}")
            row=dict(model=label.upper(),params=count,weights=str(weight),weights_sha256=digest,
                     strict_load=True,timing={},input_samples=n,
                     msemg_settings=dict(feats=64,n_layer=1,in_channels=64) if label=="msemg" else None)
            if label.startswith("sdemg"):
                timesteps=int(sdemg_cfg["sampling_steps"])
                steps=int(sdemg_cfg.get("denoise_timesteps") or timesteps)
                if not 1<=steps<=timesteps:
                    raise ValueError("Invalid SDEMG reverse-diffusion step count")
                diffusion=sdemg_classes[1](model,seq_length=n,timesteps=timesteps,
                    objective=sdemg_cfg["objective"],beta_schedule=sdemg_cfg["beta_schedule"],
                    loss_function=sdemg_cfg["loss_function"],condition=True).eval().requires_grad_(False)
                row["sdemg_settings"]=dict(feats=feats,timesteps=timesteps,denoise_steps=steps,
                    objective=sdemg_cfg["objective"],beta_schedule=sdemg_cfg["beta_schedule"],
                    sampler="original GaussianDiffusion1D.denoise; full loop including random noise",
                    params_scope="all registered ConditionalModel parameters, including original unused final stream HNF blocks")
                row["complexity_notes"]="Conv/Linear costs are counted across the actual full sampling loop; noise/update/norm/activation ops are not included. Timing covers them."
            result["cases"].append(row)
            reference=None
            for device in devices:
                print(f"[stage] {label.upper()} {device}: preparing",flush=True)
                model.to(device)
                if label=="msemg":
                    teacher_module.set_model_device_mode(model,device)
                    model.requires_grad_(False)
                    graph=MSEMGWaveform(model).eval()
                    mixers=[m for m in model.modules() if isinstance(m,teacher_module.Mamba)]
                    if any(bool(m.use_fast_path)!=(device.type=="cuda") for m in mixers):
                        raise ValueError("Mamba backend mismatch")
                elif label.startswith("sdemg"):
                    diffusion.to(device)
                    graph=SDEMGWaveform(diffusion,steps).eval()
                else:
                    graph=model
                native=wave.to(device)
                with torch.inference_mode():
                    output=graph(native)
                    if output.shape!=native.shape or not torch.isfinite(output).all():
                        raise ValueError(f"Invalid {label} waveform output: {tuple(output.shape)}")
                if "complexity" not in row:
                    if label=="fcn":
                        counted=common.count_main_ops(graph,native)
                        row["complexity"]=dict(conv_linear_gflops=counted["conv_linear_macs"]*2/1e9,
                            selective_scan_gflops_estimate=0.,layer_counts=counted["layers"])
                    elif label=="msemg":
                        row["complexity"]=costs.count_teacher_ops(graph,native,teacher_module.Mamba,common.convolution_macs)
                    else:
                        row["complexity"]=count_sdemg_ops(graph,native,common)
                    print("[complexity] "+json.dumps(dict(model=label,**{k:v for k,v in row["complexity"].items() if k!="layer_counts"})),flush=True)
                runs=args.gpu_runs if device.type=="cuda" else args.cpu_runs
                note="; CPU reference scan may take minutes" if label=="msemg" and device.type=="cpu" else (
                    f"; each run executes all {steps} reverse-diffusion steps" if label.startswith("sdemg") else "")
                print(f"[stage] {label.upper()} {device}: warmup={args.warmup}, runs={runs}{note}",flush=True)
                timing=common.measure(lambda:graph(native),device,args.warmup,runs,n/1000)
                row["timing"][device.type]=timing
                for name,value in model.state_dict().items():
                    if not torch.equal(value.detach().cpu(),original[name]):
                        raise ValueError(f"In-memory weight/buffer changed: {label}/{name}")
                if reference is not None and not label.startswith("sdemg"):
                    row["cpu_gpu_waveform_relative_l2_informational"]=float((output.cpu()-reference).norm()/reference.norm().clamp_min(1e-20))
                reference=output.detach().cpu()
                print(f"[DEVICE RESULT] {label.upper()} {device.type}: params={count}, native-waveform time={timing['median_ms']:.3f} ms",flush=True)
            del model,graph
            if label.startswith("sdemg"):
                del diffusion
        for path,digest in {**hashes,**checkpoint_hashes}.items():
            if common.file_hash(path)!=digest:
                raise ValueError(f"Source/checkpoint changed during benchmark: {path}")
        result["status"]="complete"
        save()
    except Exception as exc:
        result["status"]="failed_or_partial"
        result["error"]=str(exc)
        if not args.output.exists():save()
        raise
    print("\n===== BASELINE EFFICIENCY SUMMARY: FP32, batch=1, offline waveform =====")
    print("Model     Params  ConvLinearGFLOPs  ScanGFLOPs_est  CPU native ms  GPU native ms")
    for row in result["cases"]:
        def ms(dev):return f"{row['timing'][dev]['median_ms']:.3f}" if dev in row["timing"] else "n/a"
        print(f"{row['model']:9s} {row['params']:9d} {row['complexity']['conv_linear_gflops']:17.6f} {row['complexity']['selective_scan_gflops_estimate']:15.6f} {ms('cpu'):>14s} {ms('cuda'):>14s}")
    print("Native waveform time is the complete model inference path; these models need no STFT/iSTFT.")
    print("Scan cost is a separate estimate, not a claim of complete FLOPs. No SNRimp test or streaming delay measurement.")
    print("SDEMG rows time full reverse diffusion; registered ConditionalModel params include original unused final stream blocks. See JSON for executed Conv/Linear params and step counts.")
    print("[BASELINE BENCHMARK PASS] no training, sources/checkpoints unchanged; result="+str(args.output),flush=True)


if __name__=="__main__":
    main(CLI)
