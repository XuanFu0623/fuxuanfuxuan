
# quick_eval_audioonly_standalone.py
import os, argparse, glob, warnings
import torch
import soundfile as sf
from typing import List, Tuple
from sgmse.model import StochasticRegenerationModel
from sgmse.data_module_vi import SpecsDataModule
from sgmse.util.inference import ensure_nyquist_bin, istft_like_dm, try_pesq, try_estoi, si_sdr


def ensure_spec_bcft(S: torch.Tensor) -> torch.Tensor:
    # unify to [B,C,F,T] complex
    if S.ndim == 2:            # [F, T]
        S = S.unsqueeze(0).unsqueeze(0)
    elif S.ndim == 3:          # [C,F,T] or [1,F,T]
        if S.size(0) in (1, 2):  # treat as C
            S = S.unsqueeze(0)   # [1,C,F,T]
        else:                    # treat as [B,F,T]
            S = S.unsqueeze(1)   # [B,1,F,T]
    elif S.ndim == 4:
        pass
    else:
        raise RuntimeError(f"Unexpected spec shape: {tuple(S.shape)}")
    if not torch.is_complex(S):
        S = S.to(torch.complex64)
    return S

import traceback
warnings.filterwarnings("ignore", category=UserWarning, module="requests")

def list_pairs(base_dir: str) -> List[Tuple[str,str]]:
    ca = sorted(glob.glob(os.path.join(base_dir, "clean_audio", "*.wav")))
    na = sorted(glob.glob(os.path.join(base_dir, "noisy_audio", "*.wav")))
    # pair by basename
    clean_map = {os.path.basename(p): p for p in ca}
    pairs = []
    for p in na:
        b = os.path.basename(p)
        if b in clean_map:
            pairs.append((clean_map[b], p))
    return pairs

def wav_to_spec(w, stft_kwargs):
    if w.ndim == 1:
        w = w[None, ...]
    x = torch.tensor(w, dtype=torch.float32)
    S = torch.stft(x, **stft_kwargs)
    # Many models expect even F after multiple downsamplings; training often dropped Nyquist.
    if S.ndim >= 2 and S.shape[-2] == stft_kwargs['n_fft']//2 + 1:
        S = S[..., :-1, :]
    return S

def _center_crop_T(S, multiple=32):
    T = S.shape[-1]
    if T % multiple == 0:
        return S
    T_new = max(multiple, (T // multiple) * multiple)
    if T_new == T:
        return S
    start = max(0, (T - T_new) // 2)
    return S[..., start:start+T_new]

def spec_to_wav(S, stft_kwargs):
    S = ensure_nyquist_bin(S, stft_kwargs)
    return istft_like_dm(S, stft_kwargs)

def run_one(model, stft_kwargs, x_spec, y_spec):
    # ensure shapes to [B,C,F,T]
    x_spec = ensure_spec_bcft(x_spec)
    y_spec = ensure_spec_bcft(y_spec)

    with torch.no_grad():
        # Feed a dummy zero visual context to bypass visual encoder when model expects it
        B, C, F, T = y_spec.shape
        dummy_context = torch.zeros((B, T, 112, 112), dtype=torch.float32, device=y_spec.device)
        out = model.forward_denoiser(y_spec, context=dummy_context)
        # debug out type
        print(f"    [debug] forward_denoiser type={type(out)}")
        # forward may return just y_hat, (y_hat, aux), or a dict
        y_hat_spec = None
        if torch.is_tensor(out):
            y_hat_spec = out
        elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
            y_hat_spec = out[0]
        elif isinstance(out, dict):
            for key in ['y_hat', 'deno', 'spec', 'output', 'y_hat_spec']:
                if key in out and torch.is_tensor(out[key]):
                    y_hat_spec = out[key]
                    break
    if y_hat_spec is None:
        raise RuntimeError(f"model.forward_denoiser did not return a tensor (type={type(out)}) — check input shapes")

    # compress dims to [F,T]
    D = y_hat_spec
    if not torch.is_tensor(D):
        raise RuntimeError(f'Unexpected output type: {type(D)}')
    while D.ndim > 3:
        D = D[0]
    if D.ndim == 3:
        D = D[0]

    # phase from noisy if D is magnitude/mask
    if (not torch.is_complex(D)) or torch.abs(torch.imag(D)).max() < 1e-6:
        mag_hat = torch.abs(D) if torch.is_complex(D) else D.to(torch.float32)
        Yc = y_spec
        while Yc.ndim > 3:
            Yc = Yc[0]
        if Yc.ndim == 3:
            Yc = Yc[0]
        Yc = ensure_nyquist_bin(Yc.to(torch.complex64), stft_kwargs)
        phase = torch.angle(Yc)
        D = mag_hat.to(phase.device) * torch.exp(1j * phase)
    else:
        D = D.to(torch.complex64)

    D = ensure_nyquist_bin(D, stft_kwargs)

    wav_clean = spec_to_wav(x_spec, stft_kwargs).detach().cpu()
    wav_noisy = spec_to_wav(y_spec, stft_kwargs).detach().cpu()
    wav_deno  = spec_to_wav(D,        stft_kwargs).detach().cpu()

    L = min(wav_clean.numel(), wav_noisy.numel(), wav_deno.numel())
    return wav_clean[:L], wav_noisy[:L], wav_deno[:L]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True)
    ap.add_argument("--ckpt_path", type=str, required=True)
    ap.add_argument("--max_items", type=int, default=500)
    ap.add_argument("--num_frames", type=int, default=256)
    ap.add_argument("--audio_only", action="store_true", help="兼容参数（忽略）：始终仅音频")
    ap.add_argument("--audiovisual", action="store_true", help="兼容参数（忽略）：始终仅音频")
    ap.add_argument("--save_wavs", action="store_true")
    ap.add_argument("--out_dir", type=str, default="quick_eval_out_audioonly")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("评测模式：仅音频 (audio-only)  — 已忽略 --audio_only/--audiovisual 兼容参数")
    print(f"device={device}")

    # Build a DataModule only to recover STFT config consistent with training
    dm = SpecsDataModule(
        base_dir=args.base_dir,
        format="custom_data",
        audiovisual=False,
        num_frames=args.num_frames,
        batch_size=1,
        num_workers=0,
    )
    dm.setup("validate")
    stft_kwargs = {
        "n_fft": int(dm.stft_kwargs["n_fft"]),
        "hop_length": int(dm.stft_kwargs["hop_length"]),
        "win_length": int(dm.stft_kwargs["win_length"]),
        "window": dm.stft_kwargs["window"],
        "center": bool(dm.stft_kwargs["center"]),
        "return_complex": True,
    }

    # Load model with audiovisual=False
    print(f"Loading checkpoint: {args.ckpt_path}")
    model: StochasticRegenerationModel = StochasticRegenerationModel.load_from_checkpoint(
        args.ckpt_path,
        strict=False,
        map_location=device,
        data_module_cls=SpecsDataModule,
        base_dir=args.base_dir,
        format="custom_data",
        audiovisual=False,
        num_frames=args.num_frames,
        batch_size=1,
    ).to(device).eval()

    pairs = list_pairs(args.base_dir)
    print(f"找到配对样本: {len(pairs)} 个")
    N = min(args.max_items, len(pairs))
    print(f"评测前 {N} 个样本")

    os.makedirs(args.out_dir, exist_ok=True)

    pesq_list, sdr_list, estoi_list = [], [], []
    fs = getattr(dm, "sample_rate", 16000)

    for i, (clean_p, noisy_p) in enumerate(pairs[:N]):
        try:
            x, _ = sf.read(clean_p, dtype="float32")
            y, _ = sf.read(noisy_p, dtype="float32")
            L = min(len(x), len(y))
            x = x[:L]; y = y[:L]
            # normalize like dataset (by noisy peak)
            norm = max(1e-8, float(abs(y).max()))
            x = x / norm; y = y / norm

            X = wav_to_spec(x, stft_kwargs).to(device)
            Y = wav_to_spec(y, stft_kwargs).to(device)
            # enforce time frames to multiple of 32 (UNet down/up symmetry)
            T = min(X.shape[-1], Y.shape[-1])
            if T < 32:
                raise RuntimeError(f'Too short after STFT: T={T}')
            X = _center_crop_T(X[..., :T], 32)
            Y = _center_crop_T(Y[..., :T], 32)
            try:
                wav_x, wav_y, wav_hat = run_one(model, stft_kwargs, X, Y)
            except Exception as ee:
                print(f"    [debug] X.shape={tuple(X.shape)}, Y.shape={tuple(Y.shape)}, complexX={torch.is_complex(X)}, complexY={torch.is_complex(Y)}")
                traceback.print_exc()
                raise

            # metrics
            pesq_v = try_pesq(fs, wav_x.numpy(), wav_hat.numpy())
            estoi_v = try_estoi(fs, wav_x.numpy(), wav_hat.numpy())
            sdr_v = float(si_sdr(torch.as_tensor(wav_hat), torch.as_tensor(wav_x)))

            pesq_list.append(pesq_v if pesq_v is not None else float("nan"))
            estoi_list.append(estoi_v if estoi_v is not None else float("nan"))
            sdr_list.append(sdr_v)

            print(f"[{i:5d}] PESQ={pesq_v:.2f} | SI-SDR={sdr_v:.2f} dB | ESTOI={estoi_v:.3f}  ({os.path.basename(clean_p)})")

            if args.save_wavs:
                base = os.path.splitext(os.path.basename(clean_p))[0]
                sf.write(os.path.join(args.out_dir, f"{base}_clean.wav"), wav_x.numpy(), fs)
                sf.write(os.path.join(args.out_dir, f"{base}_noisy.wav"), wav_y.numpy(), fs)
                sf.write(os.path.join(args.out_dir, f"{base}_deno.wav"),  wav_hat.numpy(), fs)

        except Exception as e:
            print(f"[{i:5d}] 推理失败：{e}")

    def _nanmean(xs):
        t = torch.tensor([x for x in xs if isinstance(x, (int, float)) and (x == x)], dtype=torch.float32)
        return float(t.mean()) if t.numel() else float("nan")

    print("—" * 48)
    print(f"[TEST SET MEAN] PESQ={_nanmean(pesq_list):.2f} | SI-SDR={_nanmean(sdr_list):.2f} dB | ESTOI={_nanmean(estoi_list):.3f}")
    print(f"结果已保存到：{args.out_dir}/" if args.save_wavs else "未保存 wav（如需保存，加 --save_wavs）")

if __name__ == "__main__":
    main()
