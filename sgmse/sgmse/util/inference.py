# sgmse/util/inference.py
# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import torch
import torch.nn.functional as F

# -------- Optional metrics deps (graceful fallback) ----------
try:
    from pesq import pesq as pesq_fn
except Exception:
    pesq_fn = None

try:
    from pystoi.stoi import stoi as stoi_fn
except Exception:
    stoi_fn = None


# ===================== STFT / iSTFT helpers =====================
def ensure_nyquist_bin(S: torch.Tensor, stft_kwargs: dict) -> torch.Tensor:
    """
    确保频谱最后一个频点（Nyquist）存在：F == n_fft//2 + 1。
    若少一列则补，若多则裁。
    形状支持 [..., F, T]，复数张量。
    """
    if S.ndim < 2:
        return S
    n_fft = int(stft_kwargs.get("n_fft", 512))
    F_target = n_fft // 2 + 1

    F_cur = S.shape[-2]
    if F_cur == F_target:
        return S
    if F_cur == F_target - 1:
        # pad one freq bin with zeros (complex)
        pad = torch.zeros(*S.shape[:-2], 1, S.shape[-1], dtype=S.dtype, device=S.device)
        return torch.cat([S, pad], dim=-2)
    # otherwise crop or pad to target
    if F_cur > F_target:
        return S[..., :F_target, :]
    else:
        pad = torch.zeros(*S.shape[:-2], F_target - F_cur, S.shape[-1], dtype=S.dtype, device=S.device)
        return torch.cat([S, pad], dim=-2)


def istft_like_dm(S: torch.Tensor, stft_kwargs: dict) -> torch.Tensor:
    """
    使用与 DataModule 相同的参数进行 iSTFT。
    S: [F, T] (complex) 或 [*, F, T] (complex)；若有多维，取第一条。
    """
    # 取第一条以避免形状歧义
    while S.ndim > 2:
        S = S[0]
    assert torch.is_complex(S), "istft_like_dm requires complex STFT input."

    n_fft = int(stft_kwargs.get("n_fft", 512))
    hop = int(stft_kwargs.get("hop_length", 160))
    win_len = int(stft_kwargs.get("win_length", n_fft))
    window = stft_kwargs.get("window", None)
    center = bool(stft_kwargs.get("center", True))

    # iSTFT 要求 F == n_fft//2 + 1
    S = ensure_nyquist_bin(S, stft_kwargs)

    # ✅ 关键修复：window 与输入 S 放在同一设备
    if isinstance(window, torch.Tensor):
        window = window.to(S.device)

    return torch.istft(
        S,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win_len,
        window=window if isinstance(window, torch.Tensor) else None,
        center=center,
        return_complex=False,
    )


# ===================== Metrics (robust wrappers) =====================
def si_sdr(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Scale-Invariant SDR in dB.
    输入 shape: [T] 或 [1, T]；内部会展平并去均值。
    """
    est = estimate.detach().float().view(-1)
    ref = reference.detach().float().view(-1)
    est = est - est.mean()
    ref = ref - ref.mean()
    if ref.pow(2).sum() < eps:
        return -np.inf
    alpha = torch.dot(est, ref) / (ref.pow(2).sum() + eps)
    e_target = alpha * ref
    e_noise = est - e_target
    num = e_target.pow(2).sum()
    den = e_noise.pow(2).sum() + eps
    return 10.0 * torch.log10(num / den + eps).item()


def try_pesq(fs: int, ref: np.ndarray, est: np.ndarray):
    """
    PESQ wrapper. 仅当 fs ∈ {8000,16000} 且依赖存在时计算。
    返回 float 或 None
    """
    if pesq_fn is None:
        return None
    if fs not in (8000, 16000):
        return None
    try:
        # PESQ 期望 float64
        v = pesq_fn(fs, ref.astype(np.float64), est.astype(np.float64), 'wb' if fs == 8000 else 'nb')
        return float(v)
    except Exception:
        return None


def try_estoi(fs: int, ref: np.ndarray, est: np.ndarray):
    """
    ESTOI via pystoi (extended=True). 返回 float 或 None
    """
    if stoi_fn is None:
        return None
    try:
        v = stoi_fn(ref.astype(np.float64), est.astype(np.float64), fs, extended=True)
        return float(v)
    except Exception:
        return None


# ===================== Core reconstruction =====================
def _pick_first(S: torch.Tensor) -> torch.Tensor:
    while S.ndim > 3:
        S = S[0]
    if S.ndim == 3:
        S = S[0]
    return S


def _as_complex(S: torch.Tensor) -> torch.Tensor:
    return S if torch.is_complex(S) else S.to(torch.complex64)


def _rebuild_with_phase(mag_hat: torch.Tensor, phase_ref: torch.Tensor) -> torch.Tensor:
    return mag_hat.to(phase_ref.device) * torch.exp(1j * torch.angle(phase_ref))


def three_way_reconstruct(
    y_deno_spec: torch.Tensor,
    y_noisy_spec: torch.Tensor,
    stft_kwargs: dict
):
    """
    给定网络输出的频域（y_deno_spec）与 noisy 频谱（y_noisy_spec），
    以三种假设重建复谱：
      1) Direct（把输出当幅度） + noisy 相位
      2) Tanh-Mask（tanh->[-1,1]→[0,1] 掩膜）* |noisy| + noisy 相位
      3) Sigmoid-Mask（把输出当掩膜的 logit）* |noisy| + noisy 相位
    返回 dict: {name: complex_spec}
    """
    y0 = _pick_first(y_noisy_spec)
    y0 = _as_complex(y0)

    d0 = _pick_first(y_deno_spec)
    # 1) Direct
    if torch.is_complex(d0):
        mag_direct = torch.abs(d0)
    else:
        mag_direct = d0.to(torch.float32).abs()
    spec_direct = _rebuild_with_phase(mag_direct, y0)

    # 2) Tanh-Mask
    mask_tanh = torch.tanh(d0.to(torch.float32))
    mask_tanh = (mask_tanh + 1.0) * 0.5
    mask_tanh = torch.clamp(mask_tanh, 0.0, 1.0)
    spec_tanh = _rebuild_with_phase(mask_tanh * torch.abs(y0), y0)

    # 3) Sigmoid-Mask
    mask_sig = torch.sigmoid(d0.to(torch.float32))
    spec_sig = _rebuild_with_phase(mask_sig * torch.abs(y0), y0)

    # 统一 Nyquist
    spec_direct = ensure_nyquist_bin(spec_direct, stft_kwargs)
    spec_tanh   = ensure_nyquist_bin(spec_tanh,   stft_kwargs)
    spec_sig    = ensure_nyquist_bin(spec_sig,    stft_kwargs)

    return {
        "direct":  spec_direct,
        "tanh":    spec_tanh,
        "sigmoid": spec_sig
    }


# ===================== Public API used by training =====================
@torch.no_grad()
def evaluate_model(
    model,
    num_eval_files: int = 1,
    *,
    spec: bool = False,
    audio: bool = False,
    discriminative: bool = True
):
    """
    训练/验证阶段调用的评估函数。
    返回: (pesq_est, si_sdr_est, estoi_est, spec_vis, audio_vis, y_den_triplet)
        - pesq_est, si_sdr_est, estoi_est: 这里定义为 noisy->clean 的基线指标（更稳定，不会是 0）
        - spec_vis: 可选的可视化频谱（这里返回 None，避免显存占用）
        - audio_vis: 可选的可视化音频（这里返回 None；如需可在此处返回）
        - y_den_triplet: (pesq_deno, si_sdr_deno, estoi_deno) —— 去噪后的真实指标
    """
    device = next(model.parameters()).device

    # ---- 获取一个验证 batch ----
    dm = None
    if hasattr(model, "data_module") and model.data_module is not None:
        dm = model.data_module
    elif hasattr(model, "datamodule") and model.datamodule is not None:
        dm = model.datamodule
    elif hasattr(model, "trainer") and model.trainer is not None:
        try:
            dm = model.trainer.datamodule
        except Exception:
            dm = None

    if dm is None:
        raise RuntimeError("evaluate_model: cannot find a DataModule from model. "
                           "Expected model.data_module / model.datamodule / model.trainer.datamodule")

    # 有的 DM 没有 sample_rate/stft_kwargs 属性；尽量从 dm 读取，不行就用默认
    fs = getattr(dm, "sample_rate", 16000)
    stft_kwargs = getattr(dm, "stft_kwargs", {
        "n_fft": 512, "hop_length": 160, "win_length": 512, "window": None, "center": True
    })

    valloader = dm.val_dataloader() if hasattr(dm, "val_dataloader") else None
    if valloader is None:
        raise RuntimeError("evaluate_model: DataModule has no val_dataloader().")

    try:
        batch = next(iter(valloader))
    except StopIteration:
        return 0.0, 0.0, 0.0, None, None, (0.0, 0.0, 0.0)

    # 兼容 (x, y) 或 (x, y, visual)
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        x_spec = batch[0].to(device)
        y_spec = batch[1].to(device)
        visual = batch[2].to(device) if (len(batch) >= 3 and batch[2] is not None) else None
    else:
        raise RuntimeError("evaluate_model: unexpected batch structure")

    # ---- 先构造基线（noisy→clean）指标 ----
    def _prep(S):
        S = _pick_first(S)
        return ensure_nyquist_bin(_as_complex(S), stft_kwargs)

    x0 = _prep(x_spec)
    y0 = _prep(y_spec)
    wav_clean = istft_like_dm(x0, stft_kwargs).detach().cpu()
    wav_noisy = istft_like_dm(y0, stft_kwargs).detach().cpu()

    L = min(wav_clean.numel(), wav_noisy.numel())
    wav_clean = wav_clean[:L]
    wav_noisy = wav_noisy[:L]

    sdr_noisy = si_sdr(wav_noisy, wav_clean)
    pesq_noisy = try_pesq(fs, wav_clean.numpy(), wav_noisy.numpy())
    estoi_noisy = try_estoi(fs, wav_clean.numpy(), wav_noisy.numpy())

    # ---- 再过模型（仅降噪分支），做三路重建取优 ----
    try:
        y_deno_spec, _ctx = model.forward_denoiser(y_spec.to(device), context=visual)
    except Exception:
        # 模型失败时，仍然返回基线指标 + 去噪三元组为 0
        return (
            float(0.0 if pesq_noisy is None else pesq_noisy),
            float(sdr_noisy),
            float(0.0 if estoi_noisy is None else estoi_noisy),
            None,
            None,
            (0.0, 0.0, 0.0),
        )

    cand_specs = three_way_reconstruct(y_deno_spec, y_spec, stft_kwargs)
    wavs = {}
    for k, cspec in cand_specs.items():
        w = istft_like_dm(cspec, stft_kwargs).detach().cpu()
        wavs[k] = w[:min(L, w.numel())]

    # 计算三路 SI-SDR
    sdr_dir  = si_sdr(wavs["direct"],  wav_clean)
    sdr_tanh = si_sdr(wavs["tanh"],    wav_clean)
    sdr_sig  = si_sdr(wavs["sigmoid"], wav_clean)
    # 选最优一路做 PESQ/ESTOI
    cands = [("direct", sdr_dir), ("tanh", sdr_tanh), ("sigmoid", sdr_sig)]
    best_name, best_sdr = max(cands, key=lambda x: x[1])
    best_wav = wavs[best_name]

    pesq_deno = try_pesq(fs, wav_clean.numpy(), best_wav.numpy())
    estoi_deno = try_estoi(fs, wav_clean.numpy(), best_wav.numpy())

    # ---- 返回值（与你原来的 validation_step 兼容）----
    # 前三项：noisy->clean 的参考指标（稳定非 0）
    # y_den_triplet：denoised->clean 的三元组（真实反映模型）
    return (
        float(0.0 if pesq_noisy is None else pesq_noisy),
        float(sdr_noisy),
        float(0.0 if estoi_noisy is None else estoi_noisy),
        None,  # 需要可视化频谱时可改这里
        None,  # 需要可视化音频时可改这里
        (
            float(0.0 if pesq_deno is None else pesq_deno),
            float(best_sdr),
            float(0.0 if estoi_deno is None else estoi_deno),
        ),
    )
