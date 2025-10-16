# quick_eval_with_model.py
# 评测脚本：从 ckpt 恢复训练时结构；统一输入形状到 [B,C,F,T] 和 [B,T,H,W]；确保与 DM 的 STFT 一致。

import os
import sys
import argparse
import warnings
from typing import Dict, Tuple, Optional

warnings.filterwarnings("ignore", category=UserWarning, module="requests")

import torch
import torch.nn.functional as F
import soundfile as sf
import inspect

from sgmse.model import StochasticRegenerationModel
from sgmse.data_module_vi import SpecsDataModule
from sgmse.util.inference import (
    ensure_nyquist_bin,
    istft_like_dm,
    try_pesq,
    try_estoi,
    si_sdr,
)

# ----------------- 工具函数 -----------------
def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, (tuple, list)):
        return type(x)(to_device(t, device) for t in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x

def get_stft_kwargs_from_dm(dm: SpecsDataModule) -> Dict:
    # 与训练保持一致
    if hasattr(dm, "stft_kwargs"):
        stft = dm.stft_kwargs
        return dict(
            n_fft=int(stft["n_fft"]),
            hop_length=int(stft["hop_length"]),
            win_length=int(stft["win_length"]),
            window=stft["window"],
            center=bool(stft["center"]),
            return_complex=True,
        )
    # 兜底（不建议走到这里）
    print("[WARN] 未从 DataModule 读到 stft_kwargs，回退默认 512/160/512。")
    return dict(
        n_fft=512, hop_length=160, win_length=512,
        window=torch.hann_window(512), center=True, return_complex=True,
    )

def ensure_spec_bcft(S: torch.Tensor) -> torch.Tensor:
    """
    统一频谱到 [B,C,F,T] 复杂张量
    支持输入 [F,T], [1,F,T], [C,F,T], [B,C,F,T]。
    """
    if S.ndim == 2:            # [F, T]
        S = S.unsqueeze(0).unsqueeze(0)
    elif S.ndim == 3:          # 可能是 [1,F,T] 或 [C,F,T] -> 统一为 [B,C,F,T]
        if S.size(0) in (1, 2):  # 视作 C 维
            S = S.unsqueeze(0)   # [1,C,F,T]
        else:                    # 视作 [B,F,T]（极少数情况）
            S = S.unsqueeze(1)   # [B,1,F,T]
    elif S.ndim == 4:
        pass
    else:
        raise RuntimeError(f"Unexpected spec shape: {tuple(S.shape)}")
    if not torch.is_complex(S):
        S = S.to(torch.complex64)
    return S

def align_visual_to_spec(V: Optional[torch.Tensor], T: int) -> Optional[torch.Tensor]:
    """
    把视觉张量统一到 [B, T, H, W]，并与谱图帧数 T 对齐：
    - 若 Tv > T：裁剪前 T 帧
    - 若 Tv < T：重复最后一帧补齐
    """
    if V is None:
        return None
    if V.ndim == 3:         # [T, H, W]
        V = V.unsqueeze(0)  # [1, T, H, W]
    elif V.ndim == 4:       # [B, T, H, W]
        pass
    else:
        raise RuntimeError(f"Unexpected visual shape: {tuple(V.shape)}")

    Tv = V.size(1)
    if Tv > T:
        V = V[:, :T]
    elif Tv < T:
        if Tv == 0:
            # 极端兜底：若没有帧，直接用 0 填
            V = torch.zeros((V.size(0), T, V.size(2), V.size(3)), dtype=V.dtype, device=V.device)
        else:
            pad = T - Tv
            last = V[:, -1:].repeat(1, pad, 1, 1)
            V = torch.cat([V, last], dim=1)
    return V.contiguous()

def spec_to_wav(S: torch.Tensor, stft_kwargs: dict) -> torch.Tensor:
    """
    频域 -> 时域（接受 [B,C,F,T] / [C,F,T] / [F,T]）
    """
    X = S
    # 压到 [F,T]
    while X.ndim > 3:
        X = X[0]
    if X.ndim == 3:
        X = X[0]
    if not torch.is_complex(X):
        X = X.to(torch.complex64)
    X = ensure_nyquist_bin(X, stft_kwargs)
    return istft_like_dm(X, stft_kwargs)

def load_pair_specs(dm: SpecsDataModule, idx: int, device) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    # data_module_vi 返回 (X, Y, V) 三元组
    X, Y, V = dm.valid_set[idx]
    X = ensure_spec_bcft(to_device(X, device))
    Y = ensure_spec_bcft(to_device(Y, device))
    # 与谱图帧数对齐视觉
    T = Y.size(-1)
    V = to_device(V, device) if V is not None else None
    V = align_visual_to_spec(V, T)
    return X, Y, V

# ----------------- 从 ckpt 恢复模型结构 -----------------
def build_model_from_ckpt(ckpt_path: str, device, *, base_dir: str, audiovisual: bool, num_frames: int) -> StochasticRegenerationModel:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"ckpt 不存在: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    # 只覆盖数据/运行相关参数，不覆盖结构参数，避免通道不匹配
    model: StochasticRegenerationModel = StochasticRegenerationModel.load_from_checkpoint(
        ckpt_path,
        strict=False,
        map_location=device,
        data_module_cls=SpecsDataModule,
        base_dir=base_dir,
        format="custom_data",
        audiovisual=audiovisual,
        num_frames=num_frames,
        batch_size=1,
        num_workers=0,
        gpus=1,
    )
    model = model.to(device).eval()
    return model

# ----------------- 推理一条样本 -----------------
def run_one(model: StochasticRegenerationModel, dm_stft_kwargs: dict,
            x_spec: torch.Tensor, y_spec: torch.Tensor, visual: Optional[torch.Tensor],
            pass_visual: bool):
    with torch.no_grad():
        if pass_visual and (visual is not None):
            # 模型的 forward_denoiser 支持 context（cross-attention）
            y_hat_spec, _ = model.forward_denoiser(y_spec, context=visual)
        else:
            y_hat_spec, _ = model.forward_denoiser(y_spec)

    wav_clean = spec_to_wav(x_spec, dm_stft_kwargs).detach().cpu()
    wav_noisy = spec_to_wav(y_spec, dm_stft_kwargs).detach().cpu()

    D = y_hat_spec
    # 压到 [F,T]
    while D.ndim > 3:
        D = D[0]
    if D.ndim == 3:
        D = D[0]
    # 若输出不是复谱，当作幅度/掩码，配合 noisy 相位重建
    if (not torch.is_complex(D)) or torch.abs(torch.imag(D)).max() < 1e-6:
        mag_hat = torch.abs(D) if torch.is_complex(D) else D.to(torch.float32)
        Yc = y_spec
        while Yc.ndim > 3:
            Yc = Yc[0]
        if Yc.ndim == 3:
            Yc = Yc[0]
        Yc = ensure_nyquist_bin(Yc.to(torch.complex64), dm_stft_kwargs)
        phase = torch.angle(Yc)
        D = mag_hat.to(phase.device) * torch.exp(1j * phase)
    else:
        D = D.to(torch.complex64)

    D = ensure_nyquist_bin(D, dm_stft_kwargs)
    wav_deno = istft_like_dm(D, dm_stft_kwargs).detach().cpu()

    L = min(wav_clean.numel(), wav_noisy.numel(), wav_deno.numel())
    return wav_clean[:L], wav_noisy[:L], wav_deno[:L]

# ----------------- 主流程 -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True, help="包含 clean_audio/noisy_audio/lip_features 的根目录")
    ap.add_argument("--ckpt_path", type=str, required=True)
    ap.add_argument("--max_items", type=int, default=500)
    ap.add_argument("--audiovisual", action="store_true")
    ap.add_argument("--num_frames", type=int, default=256)
    ap.add_argument("--save_wavs", action="store_true")
    ap.add_argument("--out_dir", type=str, default="quick_eval_out")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # DataModule 仅用于提供验证集与 STFT 配置（和训练保持一致）
    dm = SpecsDataModule(
        base_dir=args.base_dir,
        format="custom_data",
        audiovisual=args.audiovisual,
        num_frames=args.num_frames,
        batch_size=1,
        num_workers=0,
    )
    dm.setup("validate")

    # 打印（尽量不访问可能不存在的属性）
    print(f"找到数据文件夹: {args.base_dir}/clean_audio")
    print(f"找到数据文件夹: {args.base_dir}/noisy_audio")
    print(f"找到数据文件夹: {args.base_dir}/lip_features")
    try:
        print(f"找到唇动目录: {getattr(dm, 'n_lip_dirs', 'NA')} 个")
        print(f"找到唇动帧文件: {getattr(dm, 'n_lip_frames', 'NA')} 个")
        print(f"找到清洁音频文件: {getattr(dm, 'n_clean', 'NA')} 个")
        print(f"找到噪声音频文件: {getattr(dm, 'n_noisy', 'NA')} 个")
    except Exception:
        pass
    print(f"设置样本数量: {len(dm.valid_set)}")

    stft_kwargs = get_stft_kwargs_from_dm(dm)

    # 从 ckpt 恢复模型结构
    model = build_model_from_ckpt(
        args.ckpt_path, device,
        base_dir=args.base_dir, audiovisual=args.audiovisual, num_frames=args.num_frames
    )

    # 是否可以喂视觉
    pass_visual = ("context" in inspect.signature(model.forward_denoiser).parameters) and args.audiovisual

    N_total = len(dm.valid_set)
    N = min(args.max_items, N_total)
    print(f"找到配对样本: {N_total} 个")
    print(f"评测前 {N} 个样本")

    os.makedirs(args.out_dir, exist_ok=True)

    pesq_list, sdr_list, estoi_list = [], [], []
    fs = getattr(dm, "sample_rate", 16000)

    for i in range(N):
        try:
            X, Y, V = load_pair_specs(dm, i, device)
            # 再次安全对齐视觉长度（若 DM 返回不同步）
            V = align_visual_to_spec(V, Y.size(-1)) if pass_visual else None

            wav_x, wav_y, wav_hat = run_one(model, stft_kwargs, X, Y, V, pass_visual)

            # NOTE: try_pesq/try_estoi: (fs, ref, est); si_sdr: (estimate, reference)
            pesq_v = try_pesq(fs, wav_x.numpy(), wav_hat.numpy())
            estoi_v = try_estoi(fs, wav_x.numpy(), wav_hat.numpy())
            sdr_v = float(si_sdr(torch.as_tensor(wav_hat), torch.as_tensor(wav_x)))

            pesq_list.append(pesq_v if pesq_v is not None else float("nan"))
            estoi_list.append(estoi_v if estoi_v is not None else float("nan"))
            sdr_list.append(sdr_v)

            print(f"[{i:5d}] PESQ={pesq_v if pesq_v is not None else 'NA':>4} | SI-SDR={sdr_v:6.2f} dB | ESTOI={estoi_v if estoi_v is not None else 'NA':>4}")

            if args.save_wavs and i < 50:
                base = f"idx{i:04d}"
                sf.write(os.path.join(args.out_dir, f"{base}_clean.wav"), wav_x.numpy(), fs)
                sf.write(os.path.join(args.out_dir, f"{base}_noisy.wav"), wav_y.numpy(), fs)
                sf.write(os.path.join(args.out_dir, f"{base}_deno.wav"), wav_hat.numpy(), fs)

        except Exception as e:
            print(f"[{i:5d}] 推理失败：{e}")

    def _nanmean(xs):
        t = torch.tensor([x for x in xs if isinstance(x, (int, float)) and (x == x)], dtype=torch.float32)
        return float(t.mean()) if t.numel() else float("nan")

    print("—" * 48)
    print(f"[TEST SET MEAN] PESQ={_nanmean(pesq_list):.2f} | SI-SDR={_nanmean(sdr_list):.2f} dB | ESTOI={_nanmean(estoi_list):.2f}")
    print(f"结果已保存到：{args.out_dir}/" if args.save_wavs else "未保存 wav（如需保存，加 --save_wavs）")

if __name__ == "__main__":
    main()
