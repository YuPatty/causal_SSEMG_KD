"""Independent 256/64 response KD + Teacher--Student waveform L1 experiment.

Copied training/GT/scheduler math from the fixed three-term 256/64 pipeline.
Only the optional time term changes the objective. Safety additions: visible
CUDA only, isolated outputs, metadata-checked resume, atomic checkpoint writes.
The inherited final partial-accumulation scaling is intentionally unchanged.
"""
import os, sys, csv, argparse
import hashlib, json, math, tempfile
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pipeline_spectrogram import _batch_mf_err
from cross_resolution_256_64 import (
    WaveformPairDataset,
    waveform_batch_to_spec,
)

ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT_DIR, 'MECG-E'))
from models.SSEMGNet import SSEMGNet
from models.StudentNet import StudentSSEMGNet

from distill_loss_v3_256_64 import (
    annealed_alpha,
    get_layer_curriculum_indices,
    get_noise_std,
)

from distill_loss_v3_256_64_timeKD import response_kd_with_time


TIME_KD_CSV_HEADER = [
    'epoch', 'train_gt', 'train_kd', 'alpha', 'val_gt', 'val_mf',
    'resp_mask', 'resp_mag', 'resp_pha', 'resp_com', 'feat_loss', 'rel_loss',
    'noise_std', 'sim_loss', 'resp_time', 'resp_time_weighted',
    'effective_time_kd', 'w_resp_time', 'lr',
]


def select_visible_cuda():
    """Respect Slurm CUDA visibility; never silently train on the CPU."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; this experiment will not fall back to CPU')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.empty(1, device=device)
    print(f"[GPU] {device}: {torch.cuda.get_device_name(device)} | "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    return device


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(obj, path):
    """Replace one complete checkpoint, avoiding partially written .pth files."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.timeKD-', suffix='.tmp', dir=folder)
    os.close(fd)
    try:
        torch.save(obj, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def experiment_signature(teacher_cfg, student_cfg, teacher_weights_path, data_root, settings):
    source_paths = {
        'pipeline': __file__,
        'time_loss': os.path.join(os.path.dirname(__file__), 'distill_loss_v3_256_64_timeKD.py'),
        'three_term_loss': os.path.join(ROOT_DIR, 'distill_loss_v3_256_64.py'),
        'waveform_helper': os.path.join(ROOT_DIR, 'cross_resolution_256_64.py'),
        'spectrogram_utils': os.path.join(ROOT_DIR, 'spectrogram_utils.py'),
        'student_model': os.path.join(ROOT_DIR, 'MECG-E', 'models', 'StudentNet.py'),
        'teacher_model': os.path.join(ROOT_DIR, 'MECG-E', 'models', 'SSEMGNet.py'),
        'validation_helper': os.path.join(ROOT_DIR, 'pipeline_spectrogram.py'),
    }
    return {
        'format': 'causal_256_64_timeKD_v1',
        'teacher_config': teacher_cfg, 'student_config': student_cfg,
        'teacher_weights_sha256': file_sha256(teacher_weights_path) if teacher_weights_path else None,
        'data_root': os.path.realpath(data_root),
        'settings': settings,
        'source_sha256': {key: file_sha256(path) for key, path in source_paths.items()},
        'torch_version': str(torch.__version__),
    }


def guard_output_paths(model_save, resume_path, log_csv, resume):
    paths = [os.path.abspath(path) for path in (model_save, resume_path, log_csv)]
    if len(set(paths)) != 3:
        raise ValueError('Best weights, resume state and CSV must use distinct paths')
    if resume:
        if not all(os.path.isfile(path) and os.path.getsize(path) > 0 for path in paths):
            raise FileNotFoundError('Resume requires this experiment\'s nonempty best weights, state and CSV')
    elif any(os.path.exists(path) for path in paths):
        raise FileExistsError('Fresh run refuses existing outputs; use a new name or resume this same experiment')
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def validate_resume_state(state, signature, log_csv):
    if state.get('experiment_metadata') != signature:
        raise ValueError('Resume metadata mismatch: do not use old KD/noKD states or change seed, '
                         'time weight, schedule, config, Teacher, data root or source files')
    with open(log_csv, newline='', encoding='utf-8') as stream:
        rows = list(csv.reader(stream))
    if not rows or rows[0] != TIME_KD_CSV_HEADER:
        raise ValueError('Resume CSV header does not belong to this time-KD experiment')
    data_rows = [row for row in rows[1:] if row]
    epochs = [int(row[0]) for row in data_rows]
    if (any(len(row) != len(TIME_KD_CSV_HEADER) for row in data_rows)
            or epochs != list(range(1, int(state['epoch']) + 1))):
        raise ValueError('CSV epochs and resume state do not agree; preserve files and inspect before resuming')


def cross_resolution_kd_loss(student, teacher_wav, student_noisy_spec, student_cfg, resp_weights):
    return response_kd_with_time(student, teacher_wav, student_noisy_spec, student_cfg, resp_weights)


def run_crossarch_distillation(teacher_cfg, student_cfg, teacher_weights_path, data_root,
                                total_epochs=30, batch_size=4, accum=4, lr=3e-4,
                                clip_grad=5.0, patience=5, log_csv='distill_crossarch_256_64_timeKD_log.csv',
                                model_save='model_weight/student_causal_256_64_timeKD.pth',
                                kd_weight_mode='annealed', fixed_alpha=0.5,
                                w_feature=0.0, w_response=1.0,
                                include_relation=False, w_relation=20.0,
                                resp_weights=None, no_kd=False,
                                anneal_schedule='linear', anneal_exp_k=5.0,
                                teacher_layer_selection='first', 
                                feature_layer_weights=None,
                                two_stage=False, two_stage_ratio=0.7,
                                use_trajectory=False,
                                teacher_ckpt_list=None,
                                trajectory_weights=None,
                                trajectory_curriculum='linear_grow',
                                # 新增參數 ──
                                use_feature_noise=False,
                                noise_std_start=0.5,
                                noise_std_end=0.0,
                                noise_schedule='linear',
                                use_layer_curriculum=False,
                                # 【本次新增】排程長度跟訓練總長度拆開；
                                # None 時自動 fallback 成 total_epochs（向下相容）
                                schedule_epochs=None,
                                # 【本次新增】Similarity-Preserving KD：不受 n_align_blocks
                                # 限制，1-block 架構也完整有效，見 distill_loss.py 說明
                                use_similarity=False,
                                w_similarity=1.0,
                                similarity_student_idx=-1,
                                similarity_teacher_idx=-1,
                                use_swa=False, swa_dir=None,
                                # 【本次新增】固定隨機性，讓重跑盡量逼近同一次訓練軌跡。
                                # None 時完全不設 seed（向下相容）。
                                seed=None,
                                # 【本次新增】training resume 支援。resume_path 指定的
                                # 檔案存的是「完整訓練狀態」（model+optimizer+scheduler+
                                # epoch+best_val+no_improve+RNG狀態），跟 model_save 存的
                                # 純 state_dict 是分開的兩個檔案 —— model_save 那份要維持
                                # inference_student.py 能直接吃的格式，不能混進其他東西。
                                # resume=True 時，如果 resume_path 存在就從那裡接續；
                                # 不存在則視同全新訓練（不報錯，方便同一支指令拿來啟動
                                # 或接續都可以，不需要另外判斷是不是第一次跑）。
                                resume=False,
                                resume_path=None,
                                # 【v5 新增】純加速選項，理論上不改變訓練結果（見下方各自註解）：
                                use_amp=False,
                                num_workers=0,
                                cudnn_benchmark=True):
    if resp_weights is None:
        resp_weights = {"mask": 3.5, "mag": 5.0, "pha": 0.0, "com": 2.3, "time": 0.25}


    w_time = float(resp_weights.get('time', 0.0))
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in resp_weights.values()):
        raise ValueError('KD weights must be finite and nonnegative')
    if w_response != 1.0:
        raise ValueError('Use w_response=1: change individual response weights explicitly')
    if no_kd and w_time != 0:
        raise ValueError('no_kd requires w_resp_time=0; time KD needs a Teacher')
    if not 0 <= fixed_alpha <= 1:
        raise ValueError('fixed_alpha must lie in [0,1]')
    if tuple(int(student_cfg['model'][k]) for k in ('n_fft', 'hop_size', 'win_size')) != (256, 64, 256):
        raise ValueError('This new experiment requires Student 256/64/256')
    if not student_cfg['model'].get('causal', False):
        raise ValueError('This experiment requires the causal Student')
    for cfg in (student_cfg, teacher_cfg):
        if cfg['model'].get('fea', 'pha') != 'pha' or cfg['model'].get('norm', False) is not False:
            raise ValueError('time-KD experiment requires fea=pha and norm=False for both models')
    if int(student_cfg['model']['sampling_rate']) != int(teacher_cfg['model']['sampling_rate']):
        raise ValueError('Teacher and Student sampling rates must match')
    if not all(v > 0 for v in (total_epochs, batch_size, accum, patience)):
        raise ValueError('epochs, batch_size, accum and patience must be positive')

    # Cross-resolution KD aligns only response targets after Teacher waveform
    # re-analysis.  Feature/relation/trajectory losses require same-grid TF
    # features and are intentionally not silently interpolated.
    if not no_kd:
        if w_feature != 0 or include_relation or use_similarity:
            raise ValueError("512/128 -> 256/64 KD supports response KD only: use --w_feature 0 and omit relation/similarity options.")
        if use_trajectory or use_feature_noise or use_layer_curriculum:
            raise ValueError("Trajectory, feature-noise, and layer-curriculum KD are not implemented for cross-resolution targets.")
        if float(resp_weights.get("pha", 0.0)) != 0.0:
            raise ValueError("Independent phase KD is disabled; complex KD already carries phase information. Set --w_resp_pha 0.")
        teacher_grid = tuple(int(teacher_cfg["model"][k]) for k in ("n_fft", "hop_size", "win_size"))
        student_grid = tuple(int(student_cfg["model"][k]) for k in ("n_fft", "hop_size", "win_size"))
        if teacher_grid != (512, 128, 512) or student_grid != (256, 64, 256):
            raise ValueError(
                "This script is for Teacher (n_fft, hop, win)=(512,128,512) "
                "and Student=(256,64,256); received "
                f"Teacher={teacher_grid}, Student={student_grid}."
            )

    # ── 固定隨機性 ──
    # 必須放在任何模型建構（StudentSSEMGNet/FeatureProjector 的權重初始化）、
    # 任何 DataLoader 建立（train_loader 的 shuffle=True）之前，這兩者都會用掉
    # torch 的全域 RNG，晚設 seed 等於沒設。不保證跟 cuDNN 非確定性 kernel
    # 100% bit-exact，但足以大幅縮小兩次重跑之間的變異。
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"[seed] 已固定 random/numpy/torch seed = {seed}")

    # 【本次新增】schedule_epochs 沒指定時，行為跟修改前完全一樣
    if schedule_epochs is None:
        schedule_epochs = total_epochs
    if schedule_epochs > total_epochs:
        print(f"[warning] --schedule_epochs ({schedule_epochs}) > --epochs ({total_epochs})，"
              f"排程永遠走不完，自動裁切成 {total_epochs}")
        schedule_epochs = total_epochs
    if schedule_epochs != total_epochs:
        print(f"[Schedule] 排程長度={schedule_epochs} epoch，訓練總長度={total_epochs} epoch。"
              f"epoch > {schedule_epochs} 之後，alpha/noise/layer curriculum 全部凍結在排程終點值，"
              f"純粹用該終點設定繼續訓練（不會再變化）。")


    if resume_path is None:
        resume_path = os.path.splitext(model_save)[0] + '.resume_state.pth'
    guard_output_paths(model_save, resume_path, log_csv, resume)
    signature = experiment_signature(teacher_cfg, student_cfg, teacher_weights_path, data_root, {
        'seed': seed, 'epochs': total_epochs, 'schedule_epochs': schedule_epochs,
        'batch_size': batch_size, 'accum': accum, 'lr': lr, 'clip_grad': clip_grad,
        'patience': patience, 'resp_weights': dict(resp_weights),
        'kd_weight_mode': kd_weight_mode, 'fixed_alpha': fixed_alpha,
        'anneal_schedule': anneal_schedule, 'anneal_exp_k': anneal_exp_k,
        'no_kd': no_kd, 'num_workers': num_workers, 'use_amp': use_amp,
        'cudnn_benchmark': cudnn_benchmark, 'two_stage': two_stage,
        'two_stage_ratio': two_stage_ratio, 'use_swa': use_swa,
    })
    print('[experiment] ' + json.dumps(signature, sort_keys=True, ensure_ascii=False))
    print(f'[timeKD] w_resp_time={w_time:g}; L1(Student waveform, stopgrad(Teacher waveform)); '
          'GT, three original KD weights and alpha schedule are unchanged')
    print('[grid] Teacher n_fft/hop/win=(512,128,512); Student=(256,64,256); causal=True')

    device = select_visible_cuda()

    # 【v5 新增】cuDNN autotuning：固定輸入 shape 的情況下，讓 cuDNN
    # 自動挑選最快的卷積演算法，不改變計算結果（同一組權重、同一筆輸入，
    # 算出來的數值應該一致），只是挑 kernel 實作的過程有極小的非確定性
    # （不同次執行可能選到不同但同樣正確的 kernel）。如果需要追求 bit-exact
    # 重現性，可以傳 cudnn_benchmark=False 關掉，但這個非確定性遠小於本專案
    # 已經在容忍的其他雜訊來源（cuDNN 本身非確定性 kernel、不同硬體等）。
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = cudnn_benchmark
        if cudnn_benchmark:
            print("[speed] 已啟用 torch.backends.cudnn.benchmark=True（固定 shape 下自動挑最快 kernel）")

    # 【v5 新增】AMP (Automatic Mixed Precision)：forward 用 FP16/BF16 加速，
    # loss/梯度累積用 GradScaler 維持 FP32 精度，不改變計算圖或 batch 組成，
    # 只是數值精度上有極小誤差（通常遠小於訓練本身的隨機性雜訊）。
    amp_dtype = torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == 'cuda'))
    if use_amp:
        if device.type == 'cuda':
            print(f"[speed] 已啟用 AMP混合精度訓練 (autocast dtype={amp_dtype})")
        else:
            print("[speed] --use_amp 有指定，但目前裝置是 CPU，AMP 只在 CUDA 上有意義，已自動略過")

    teacher = None
    projector = None

    if not no_kd:
        teacher = SSEMGNet(teacher_cfg).to(device)
        teacher.load_state_dict(torch.load(teacher_weights_path, map_location=device, weights_only=False))
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    student = StudentSSEMGNet(student_cfg).to(device)

    n_params_s = sum(p.numel() for p in student.parameters())
    if teacher is not None:
        n_params_t = sum(p.numel() for p in teacher.parameters())
        print(f"[CrossKD] Teacher(Mamba) params: {n_params_t/1e6:.2f}M | "
              f"Student(Conv) params: {n_params_s/1e6:.2f}M | "
              f"Compression: {n_params_t/max(1,n_params_s):.2f}x")
    else:
        print(f"[NoKD baseline] Student(Conv) params: {n_params_s/1e6:.2f}M "
              f"| training WITHOUT teacher, GT loss only")

    trainable_params = list(student.parameters())

    optim = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.8, 0.99), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.5, patience=2, min_lr=1e-6
    )

    # 【本次新增】resume_path 預設值：跟 model_save 放一起，但檔名不同，
    # 避免跟 inference_student.py 直接讀取的那份純 state_dict 混在一起。
    if resume_path is None:
        base = model_save[:-4] if model_save.endswith('.pth') else model_save
        resume_path = base + '.resume_state.pth'

    start_epoch = 1
    best_val = float('inf')
    no_improve = 0
    best_epoch = None
    resumed = False

    if resume and os.path.exists(resume_path):
        print(f"[resume] 找到 {resume_path}，從中接續訓練狀態...")
        state = torch.load(resume_path, map_location=device, weights_only=False)
        validate_resume_state(state, signature, log_csv)
        best_epoch = state.get('best_epoch')
        student.load_state_dict(state['student_state_dict'])
        if projector is not None and state.get('projector_state_dict') is not None:
            projector.load_state_dict(state['projector_state_dict'])
        optim.load_state_dict(state['optim_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        if use_amp and state.get('scaler_state_dict') is not None:
            scaler.load_state_dict(state['scaler_state_dict'])
        best_val = state['best_val']
        no_improve = state['no_improve']
        start_epoch = state['epoch'] + 1
        # 還原 RNG 狀態，讓接續後的隨機性（DataLoader shuffle 順序等）
        # 盡量接續上次中斷的軌跡，而不是從一個新的隨機狀態重新開始。
        if 'torch_rng_state' in state:
            torch.set_rng_state(state['torch_rng_state'].cpu())
        if device.type == 'cuda' and 'cuda_rng_state' in state and state['cuda_rng_state'] is not None:
            torch.cuda.set_rng_state(state['cuda_rng_state'].cpu(), device=device)
        if 'numpy_rng_state' in state and state['numpy_rng_state'] is not None:
            import numpy as np
            np.random.set_state(state['numpy_rng_state'])
        if 'python_rng_state' in state and state['python_rng_state'] is not None:
            import random
            random.setstate(state['python_rng_state'])
        resumed = True
        print(f"[resume] 已還原到 epoch {state['epoch']} 結束時的狀態，"
              f"從 epoch {start_epoch} 繼續（best_val={best_val:.4f}, no_improve={no_improve}）")
    elif resume:
        raise FileNotFoundError(f'Resume state disappeared before loading: {resume_path}')

    # Keep waveforms in the Dataset.  Each batch is analysed twice below: once
    # on the Teacher's 512/128 grid and once on the Student's 256/64 grid.
    train_ds = WaveformPairDataset(data_root, 'train')
    valid_ds = WaveformPairDataset(data_root, 'valid')
    # 【v5 修改】原本寫死 num_workers=0，改成吃參數。這一項純粹是「資料怎麼
    # 餵進去」的機制（平行預取 vs. 主 process 同步準備），不影響任何梯度
    # 計算或 batch 內樣本組成，理論上不改變訓練結果。因為現在資料是逐筆
    # 即時算 STFT（不是一次性整包 load 進記憶體），num_workers>0 的效益
    # 比 v5 之前更明顯，建議實際設一個 >0 的值。
    use_pin = (device.type == 'cuda')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers, pin_memory=use_pin,
                              persistent_workers=(num_workers > 0))
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              drop_last=False, num_workers=num_workers, pin_memory=use_pin,
                              persistent_workers=(num_workers > 0))

    os.makedirs(os.path.dirname(model_save), exist_ok=True)

    # 【本次新增】SWA checkpoint 資料夾初始化
    if use_swa:
        if swa_dir is None:
            base = model_save[:-4] if model_save.endswith('.pth') else model_save
            swa_dir = base + '_swa_ckpts'
        os.makedirs(swa_dir, exist_ok=True)
        print(f"[SWA] 排程走完後每個 epoch 的 checkpoint 會存到 {swa_dir}/")

    # 【本次修改】resume 時用 append 模式接續寫，不要清空已有的訓練曲線；
    # 全新訓練（或 resume_path 不存在）才用 'w' 從頭寫入含表頭的新檔案。
    if resumed:
        log_csv_mode = 'a'
        print(f"[resume] {log_csv} 用 append 模式接續寫入，不會清空先前紀錄")
    else:
        log_csv_mode = 'w'
    if log_csv_mode == 'w':
        with open(log_csv, 'w', newline='') as f:
            csv.writer(f).writerow(TIME_KD_CSV_HEADER)

    # 預先取得 teacher 總層數，供層級課程使用
    teacher_total_layers = teacher_cfg['model']['num_tscblocks'] if not no_kd else 4

    if resumed and no_improve >= patience:
        print(f"[resume] 還原後的 no_improve({no_improve}) 已經 >= patience({patience})，"
              f"代表上次中斷前其實已經該觸發 early stopping 了，這裡直接結束、不再繼續訓練。")
        return

    for ep in range(start_epoch, total_epochs + 1):
        student.train()
        if projector is not None:
            projector.train()
        running_gt, running_kd = 0.0, 0.0
        running_info = {"resp_mask": 0.0, "resp_mag": 0.0, "resp_pha": 0.0,
                        "resp_com": 0.0, "feat_loss": 0.0, "rel_loss": 0.0,
                        "sim_loss": 0.0, "resp_time": 0.0, "resp_time_weighted": 0.0}

        # 【本次修改】排程用的「有效 epoch」封頂在 schedule_epochs，
        # 超過之後就一直固定用 schedule_epochs 那個值去算，
        # 等同於凍結在排程終點（alpha=0、noise=noise_std_end、layer curriculum 全開）。
        sched_ep = min(ep, schedule_epochs)

        # ── 計算當前 noise_std ──
        if use_feature_noise:
            cur_noise_std = get_noise_std(sched_ep, schedule_epochs, noise_std_start,
                                          noise_std_end, noise_schedule)
        else:
            cur_noise_std = 0.0

        # ── 計算當前層級對齊索引 ──
        if use_layer_curriculum:
            all_layers = list(range(teacher_total_layers))
            cur_teacher_indices = get_layer_curriculum_indices(sched_ep, schedule_epochs, all_layers)
        else:
            cur_teacher_indices = teacher_layer_selection

        mode_tag = "NoKD" if no_kd else (
            "TrajKD" if use_trajectory else (
                f"CrossKD:{kd_weight_mode}" + ("+noise" if use_feature_noise else "") +
                ("+layerCur" if use_layer_curriculum else "")
            )
        )
        bar = tqdm(train_loader, desc=f"[{mode_tag}] epoch {ep}/{total_epochs}")

        for step, (clean_wav, noisy_wav) in enumerate(bar, start=1):
            clean_wav = clean_wav.to(device, non_blocking=True)
            noisy_wav = noisy_wav.to(device, non_blocking=True)
            teacher_clean, teacher_noisy = waveform_batch_to_spec(clean_wav, noisy_wav, teacher_cfg)
            student_clean, student_noisy = waveform_batch_to_spec(clean_wav, noisy_wav, student_cfg)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(use_amp and device.type == 'cuda')):
                gt_loss = student(student_clean, student_noisy)

                if no_kd:
                    total_loss = gt_loss
                    kd_loss = torch.tensor(0.0, device=device)
                    alpha = 0.0
                    info = {}
                elif two_stage:
                    phase1_epochs = int(total_epochs * two_stage_ratio)
                    if ep <= phase1_epochs:
                        with torch.no_grad():
                            teacher(teacher_clean, teacher_noisy)
                        with torch.autocast(device_type=device.type, enabled=False):
                            kd_loss, info = cross_resolution_kd_loss(
                                student, teacher.last_wavs["pred"], student_noisy, student_cfg, resp_weights)
                        total_loss = kd_loss
                        alpha = 1.0
                    else:
                        total_loss = gt_loss
                        kd_loss = torch.tensor(0.0, device=device)
                        alpha = 0.0
                        info = {}
                elif use_trajectory and teacher_state_dicts is not None:
                    active_idx = get_trajectory_active_indices(
                        sched_ep, schedule_epochs, len(teacher_state_dicts),
                        mode=trajectory_curriculum
                    )
                    traj_kd, _ = compute_trajectory_kd_batch(
                        student, teacher, teacher_state_dicts,
                        clean_b, noisy_b,
                        projector,
                        trajectory_weights=trajectory_weights,
                        active_indices=active_idx,
                        w_feature=w_feature, w_response=w_response,
                        include_relation=include_relation,
                        w_relation=w_relation,
                        resp_weights=resp_weights,
                        teacher_layer_selection=cur_teacher_indices,
                        feature_layer_weights=feature_layer_weights,
                        device=device,
                        noise_std=cur_noise_std)
                    if kd_weight_mode == 'annealed':
                        alpha = annealed_alpha(sched_ep, schedule_epochs,
                                               schedule=anneal_schedule, exp_k=anneal_exp_k)
                    else:
                        alpha = fixed_alpha
                    kd_loss = traj_kd
                    total_loss = (1.0 - alpha) * gt_loss + alpha * kd_loss
                    info = {"kd_loss": float(kd_loss.detach()), "alpha": alpha, "noise_std": cur_noise_std}
                else:
                    # 傳統單一 teacher + 混合蒸餾
                    # 【優化】排程已凍結在終點（sched_ep >= schedule_epochs）且 annealed alpha
                    # 已經衰減到 0 時，KD loss 對 total_loss 的貢獻恆為 0，這時完全不需要
                    # 再做 teacher forward + KD loss 計算，直接退化成純 GT loss 微調，
                    # 省下 phase 2（例如 61~150 epoch）整段的 teacher 推論成本。
                    # 只在 annealed 模式下判斷；fixed alpha 模式下 alpha 通常不為 0，維持原行為。
                    skip_kd_this_step = False
                    if kd_weight_mode == 'annealed':
                        alpha_probe = annealed_alpha(sched_ep, schedule_epochs,
                                                     schedule=anneal_schedule, exp_k=anneal_exp_k)
                        if sched_ep >= schedule_epochs and alpha_probe <= 1e-8:
                            skip_kd_this_step = True

                    if skip_kd_this_step:
                        total_loss = gt_loss
                        kd_loss = torch.tensor(0.0, device=device)
                        alpha = 0.0
                        info = {}
                    else:
                        with torch.no_grad():
                            teacher(teacher_clean, teacher_noisy)
                        if kd_weight_mode == 'annealed':
                            alpha = annealed_alpha(sched_ep, schedule_epochs,
                                                   schedule=anneal_schedule, exp_k=anneal_exp_k)
                        else:
                            alpha = fixed_alpha
                        with torch.autocast(device_type=device.type, enabled=False):
                            kd_loss, info = cross_resolution_kd_loss(
                                student, teacher.last_wavs["pred"], student_noisy, student_cfg, resp_weights)
                        total_loss = (1.0 - alpha) * gt_loss + alpha * kd_loss

            if step == 1 and ep == start_epoch and info.get('resp_time_weighted') is not None:
                print(f"[timeKD] raw={info['resp_time']:.8g} | weighted={info['resp_time_weighted']:.8g} "
                      f"| alpha*weighted={alpha * info['resp_time_weighted']:.8g}")
            if not torch.isfinite(total_loss):
                raise FloatingPointError('Non-finite total training loss; no checkpoint will be saved')

            # 【v5 修改】AMP 下用 GradScaler 縮放 loss 再 backward，避免 FP16 梯度
            # underflow；不啟用 AMP 時 scaler 是 disabled 狀態，scale(x)==x，
            # 行為等同原本直接呼叫 .backward()，不影響非 AMP 路徑。
            scaler.scale(total_loss / accum).backward()

            if step % accum == 0 or step == len(train_loader):
                # clip_grad_norm_ 之前要先 unscale_，否則量到的梯度範數是被
                # scaler 放大過的，clip 的門檻會不對。
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=clip_grad)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            running_gt += float(gt_loss.detach())
            running_kd += float(kd_loss.detach())
            for k in running_info:
                v = info.get(k)
                running_info[k] += (v if v is not None else 0.0)
            bar.set_postfix(gt=f"{running_gt/step:.4g}", kd=f"{running_kd/step:.4g}",
                            alpha=f"{alpha:.2f}", noise=f"{cur_noise_std:.2f}")

        student.eval()
        val_gt, val_mf_sum, n_val_samples, n_batches = 0.0, 0.0, 0, 0
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype,
                                              enabled=(use_amp and device.type == 'cuda')):
            for clean_wav, noisy_wav in valid_loader:
                clean_wav = clean_wav.to(device, non_blocking=True)
                noisy_wav = noisy_wav.to(device, non_blocking=True)
                student_clean, student_noisy = waveform_batch_to_spec(clean_wav, noisy_wav, student_cfg)
                gt_loss = student(student_clean, student_noisy)
                val_gt += float(gt_loss.detach())
                n_batches += 1
                last_wavs = getattr(student, "last_wavs", None)
                if last_wavs is not None:
                    mf_err = _batch_mf_err(last_wavs["pred"], last_wavs["clean"],
                                            student_cfg.get('metrics', {}))
                    val_mf_sum += mf_err * last_wavs["pred"].shape[0]
                    n_val_samples += last_wavs["pred"].shape[0]

        val_gt /= max(1, n_batches)
        val_mf = (val_mf_sum / n_val_samples) if n_val_samples > 0 else float('nan')
        train_gt_epoch = running_gt / max(1, len(train_loader))
        train_kd_epoch = running_kd / max(1, len(train_loader))

        mean_time = running_info['resp_time'] / max(1, len(train_loader))
        mean_weighted_time = running_info['resp_time_weighted'] / max(1, len(train_loader))
        lr_used = optim.param_groups[0]['lr']

        print(f"★ Epoch {ep} | train_gt={train_gt_epoch:.4f} | train_kd={train_kd_epoch:.4f} "
              f"| alpha={alpha:.2f} | noise={cur_noise_std:.3f} | val_gt={val_gt:.4f} | val_mf={val_mf:.4f} Hz "
              f"| time_kd={mean_time:.7g} | time_eff={alpha * mean_weighted_time:.7g} | lr={lr_used:.3g}")

        with open(log_csv, 'a', newline='') as f:
            row = [ep, train_gt_epoch, train_kd_epoch, alpha, val_gt, val_mf]
            row += [running_info[k] / max(1, len(train_loader))
                    for k in ["resp_mask", "resp_mag", "resp_pha", "resp_com", "feat_loss", "rel_loss"]]
            row += [cur_noise_std]
            row += [running_info["sim_loss"] / max(1, len(train_loader))]
            row += [mean_time, mean_weighted_time, alpha * mean_weighted_time, w_time, lr_used]
            csv.writer(f).writerow(row)

        scheduler.step(val_gt)

        is_phase_1 = (two_stage and ep <= int(total_epochs * two_stage_ratio))

        if val_gt < best_val:
            best_val = val_gt
            best_epoch = ep
            no_improve = 0
            atomic_torch_save(student.state_dict(), model_save)
            print(f"✓ new best student (val_gt={best_val:.4f}) saved to {model_save}")
        else:
            if not is_phase_1:
                no_improve += 1
                print(f"✗ no improvement ({no_improve}/{patience})")
            else:
                print(f"ℹ [Phase 1] val_gt plateau expected")

        # 【本次新增】排程已走完（sched_ep >= schedule_epochs）時，
        # 每個 epoch 都額外存一份 checkpoint，供訓練結束後做 SWA 平均。
        # 不影響上面 best-checkpoint 的邏輯，純粹多存一份。
        if use_swa and sched_ep >= schedule_epochs:
            swa_ckpt_path = os.path.join(swa_dir, f"epoch{ep}_valgt{val_gt:.5f}.pth")
            atomic_torch_save(student.state_dict(), swa_ckpt_path)

        # 【本次新增】每個 epoch 結束都存一份「完整訓練狀態」供 resume 用，
        # 覆蓋寫同一個檔案（不像 best checkpoint 或 SWA checkpoint 那樣要保留
        # 歷史版本 —— resume 只需要「最新一份能接續的狀態」）。
        # 放在 best-checkpoint 判斷之後，這樣 state 裡的 best_val/no_improve
        # 才是這個 epoch 結束當下的最新值。
        resume_state = {
            'epoch': ep,
            'best_epoch': best_epoch,
            'experiment_metadata': signature,
            'student_state_dict': student.state_dict(),
            'projector_state_dict': (projector.state_dict() if projector is not None else None),
            'optim_state_dict': optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': (scaler.state_dict() if use_amp else None),
            'best_val': best_val,
            'no_improve': no_improve,
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': (torch.cuda.get_rng_state(device) if device.type == 'cuda' else None),
        }
        try:
            import numpy as np
            resume_state['numpy_rng_state'] = np.random.get_state()
        except Exception:
            resume_state['numpy_rng_state'] = None
        try:
            import random
            resume_state['python_rng_state'] = random.getstate()
        except Exception:
            resume_state['python_rng_state'] = None
        atomic_torch_save(resume_state, resume_path)

        if not is_phase_1 and no_improve >= patience:
            print(f"⏹ Early stopping at epoch {ep}")
            break


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--teacher_config', required=True)
    p.add_argument('--teacher_weights', default=None)
    p.add_argument('--student_config', required=True)
    p.add_argument('--data_root', default='semg_data/processed',
                   help='waveform data_root（prepare_data.py 的輸出，內含 '
                        'train/valid/{noisy,clean}/*.npy）；不再需要先跑 '
                        'convert.py / make_dataset_spectrogram.py 產生 .pt 快取')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--schedule_epochs', type=int, default=None,
                   help='alpha/noise/layer curriculum 排程長度；預設等於 --epochs（向下相容）。'
                        '設成比 --epochs 小的值，可以讓排程照原節奏走完後，'
                        '用排程終點設定繼續訓練，不影響已驗證過的排程動態。')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--accum', type=int, default=4)
    p.add_argument('--no_kd', action='store_true')
    p.add_argument('--kd_weight_mode', choices=['fixed', 'annealed'], default='annealed')
    p.add_argument('--fixed_alpha', type=float, default=0.5)
    p.add_argument('--anneal_schedule', choices=['linear', 'cosine', 'exponential'], default='linear')
    p.add_argument('--anneal_exp_k', type=float, default=5.0)
    p.add_argument('--w_feature', type=float, default=0.0,
                   help='Cross-resolution KD only supports response loss; must remain 0.')
    p.add_argument('--w_response', type=float, default=1.0)
    p.add_argument('--include_relation', action='store_true')
    p.add_argument('--w_relation', type=float, default=20.0)
    p.add_argument('--teacher_indices', type=int, nargs='+', default=None)
    p.add_argument('--w_resp_mask', type=float, default=3.5)
    p.add_argument('--w_resp_mag', type=float, default=5.0)
    p.add_argument('--w_resp_pha', type=float, default=0.0,
                   help='Keep at 0: complex KD already provides phase supervision.')
    p.add_argument('--w_resp_com', type=float, default=2.3)
    p.add_argument('--w_resp_time', type=float, default=0.25,
                   help='Waveform L1 KD weight; 0 disables it exactly. 0.25 is an untuned trial, not an optimum.')
    p.add_argument('--log_csv', default='distill_crossarch_256_64_timeKD_log.csv')
    p.add_argument('--model_save', default='model_weight/student_causal_256_64_timeKD.pth')
    p.add_argument('--feature_layer_weights', type=float, nargs='+', default=None)
    p.add_argument('--two_stage', action='store_true')
    p.add_argument('--two_stage_ratio', type=float, default=0.7)
    p.add_argument('--patience', type=int, default=10, help='early stopping patience')

    # 【本次新增】SWA（Stochastic Weight Averaging）checkpoint 收集
    p.add_argument('--swa', action='store_true',
                   help='排程走完後（sched_ep >= schedule_epochs）每個 epoch 額外存一份 '
                        'checkpoint 到 --swa_dir，供訓練結束後做權重平均。不影響原本 '
                        'best-checkpoint 的存法，是額外多存，兩者互不影響。')
    p.add_argument('--swa_dir', default=None,
                   help='SWA checkpoint 存放資料夾，預設是 model_save 去掉副檔名 + "_swa_ckpts"')

    # Trajectory 參數（原功能保留）
    p.add_argument('--trajectory', action='store_true')
    p.add_argument('--teacher_ckpt_list', nargs='+', default=None)
    p.add_argument('--trajectory_weights', nargs='+', type=float, default=None)
    p.add_argument('--trajectory_curriculum', choices=['all', 'linear_grow'], default='linear_grow')

    # 新增：退化教師蒸餾（加噪衰減）
    p.add_argument('--feature_noise', action='store_true',
                   help='對 teacher 中間層特徵加噪，並隨 epoch 衰減（模擬 Teacher Trajectory）')
    p.add_argument('--noise_std_start', type=float, default=0.5,
                   help='初始噪聲標準差（預設 0.5）')
    p.add_argument('--noise_std_end', type=float, default=0.0,
                   help='最終噪聲標準差（預設 0.0）')
    p.add_argument('--noise_schedule', choices=['linear', 'cosine', 'exponential'], default='linear',
                   help='噪聲衰減曲線（同 alpha 排程）')

    # 新增：層級課程
    p.add_argument('--layer_curriculum', action='store_true',
                   help='啟用層級課程：前期只用淺層 teacher，逐步加入深層')

    # 【本次新增】Similarity-Preserving KD：不受 n_align_blocks 限制，
    # 學生只有 1 個 block 時也完整有效（見 distill_loss.py 說明）
    p.add_argument('--similarity_preserving', action='store_true',
                   help='啟用 Similarity-Preserving KD（batch 內樣本兩兩相似度矩陣對齊，'
                        '不需要 FeatureProjector、不受 block 數限制）')
    p.add_argument('--w_similarity', type=float, default=1.0,
                   help='Similarity-Preserving loss 權重（預設 1.0，未經調參，建議自行掃描）')
    p.add_argument('--similarity_student_idx', type=int, default=-1,
                   help='取學生 last_feats 的第幾層算相似度矩陣，預設 -1（最後一層）')
    p.add_argument('--similarity_teacher_idx', type=int, default=-1,
                   help='取 teacher 全部 feats 的第幾層算相似度矩陣，預設 -1（最深層）')
    p.add_argument('--seed', type=int, default=None,
                   help='固定 random/numpy/torch 的隨機種子，讓重跑盡量逼近同一次訓練軌跡。'
                        '不指定時完全不設 seed(向下相容)。GPU 上某些 cuDNN kernel 本身非'
                        '完全確定性，設了 seed 只能大幅縮小、不能保證 100% 消除重跑間的差異。')

    # 【本次新增】training resume
    p.add_argument('--resume', action='store_true',
                   help='Resume only this time-KD run: best weights, state and CSV must exist; '
                        'matching experiment metadata is required. Never use an old three-term/noKD state.')
    p.add_argument('--resume_path', default=None,
                   help='resume 用的完整訓練狀態檔案路徑；不指定時預設是 '
                        '--model_save 去掉副檔名 + ".resume_state.pth"')

    # 【v5 新增】純加速選項，理論上不改變訓練結果
    p.add_argument('--use_amp', action='store_true',
                   help='啟用 AMP 混合精度訓練 (FP16 autocast + GradScaler)。'
                        '不改變計算圖或 batch 組成，只是數值精度略有誤差，'
                        '通常遠小於訓練本身的隨機性雜訊。只在 CUDA 上有效果，'
                        'CPU 上會自動略過。')
    p.add_argument('--num_workers', type=int, default=0,
                   help='DataLoader 平行預取的 worker 數量，預設 0（沿用舊行為）。'
                        '純粹是資料怎麼餵進去的機制，不影響訓練結果。因為資料是'
                        '一次性整包 load 進記憶體，效益可能有限，但改了無害。')
    p.add_argument('--no_cudnn_benchmark', action='store_true',
                   help='關閉 torch.backends.cudnn.benchmark（預設是開啟的）。'
                        '如果需要追求 bit-exact 重現性可以關掉，但這個非確定性'
                        '通常遠小於本專案已經在容忍的其他雜訊來源。')

    args = p.parse_args()

    if not args.no_kd and not args.trajectory and args.teacher_weights is None:
        p.error("必須指定 --teacher_weights（或使用 --no_kd 或 --trajectory）")

    with open(args.teacher_config) as f:
        teacher_cfg = yaml.safe_load(f)
    with open(args.student_config) as f:
        student_cfg = yaml.safe_load(f)

    default_log = 'log_student_noKD_causal_256_64_timeKD_control.csv' if args.no_kd else 'distill_crossarch_256_64_timeKD_log.csv'
    default_save = 'model_weight/student_noKD_causal_256_64_timeKD_control.pth' if args.no_kd else 'model_weight/student_causal_256_64_timeKD.pth'
    log_csv = args.log_csv if args.log_csv != 'distill_crossarch_256_64_timeKD_log.csv' else default_log
    model_save = args.model_save if args.model_save != 'model_weight/student_causal_256_64_timeKD.pth' else default_save

    layer_selection = args.teacher_indices if args.teacher_indices is not None else 'first'

    run_crossarch_distillation(
        teacher_cfg, student_cfg, args.teacher_weights, args.data_root,
        total_epochs=args.epochs, batch_size=args.batch_size, accum=args.accum,
        kd_weight_mode=args.kd_weight_mode, fixed_alpha=args.fixed_alpha,
        w_feature=args.w_feature, w_response=args.w_response,
        include_relation=args.include_relation, w_relation=args.w_relation,
        log_csv=log_csv, model_save=model_save,
        resp_weights={"mask": args.w_resp_mask, "mag": args.w_resp_mag,
                      "pha": args.w_resp_pha, "com": args.w_resp_com,
                      "time": args.w_resp_time},
        no_kd=args.no_kd, anneal_schedule=args.anneal_schedule, anneal_exp_k=args.anneal_exp_k,
        teacher_layer_selection=layer_selection,
        feature_layer_weights=args.feature_layer_weights,
        two_stage=args.two_stage, two_stage_ratio=args.two_stage_ratio,
        use_trajectory=args.trajectory,
        teacher_ckpt_list=args.teacher_ckpt_list,
        trajectory_weights=args.trajectory_weights,
        trajectory_curriculum=args.trajectory_curriculum,
        use_feature_noise=args.feature_noise,
        noise_std_start=args.noise_std_start,
        noise_std_end=args.noise_std_end,
        noise_schedule=args.noise_schedule,
        use_layer_curriculum=args.layer_curriculum,
        patience=args.patience,
        schedule_epochs=args.schedule_epochs,
        use_similarity=args.similarity_preserving,
        w_similarity=args.w_similarity,
        similarity_student_idx=args.similarity_student_idx,
        similarity_teacher_idx=args.similarity_teacher_idx,
        use_swa=args.swa,
        swa_dir=args.swa_dir,
        seed=args.seed,
        resume=args.resume,
        resume_path=args.resume_path,
        use_amp=args.use_amp,
        num_workers=args.num_workers,
        cudnn_benchmark=(not args.no_cudnn_benchmark)
    )
