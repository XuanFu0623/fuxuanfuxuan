import os
from glob import glob
from os.path import join

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import librosa
import pytorch_lightning as pl

# -------------------------------
# Utils
# -------------------------------
SEED = 20
np.random.seed(SEED)


def get_window(window_type: str, window_length: int):
    if window_type == "sqrthann":
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    if window_type == "hann":
        return torch.hann_window(window_length, periodic=True)
    raise NotImplementedError(f"Window type {window_type} not implemented!")


# -------------------------------
# Dataset
# -------------------------------
class Specs(Dataset):
    """Dataset for FlowAVSE minimal bring‑up with per‑frame visual alignment.

    Directory layout (custom_data):
        base_dir/
          ├─ clean_audio/*.wav
          ├─ noisy_audio/*.wav
          └─ lip_features/<sample_name>/
               ├─ 00000.pt | .npy | .jpg | .png
               ├─ 00001.pt | ...
               └─ ...
    """

    def __init__(
        self,
        data_dir: str,
        subset: str,
        dummy: bool,
        shuffle_spec: bool,
        num_frames: int,
        format: str,
        use_sync_encoder: bool,  # kept for interface compatibility
        *,
        audiovisual: bool = True,
        normalize_audio: bool = False,
        spec_transform=None,
        stft_kwargs=None,
        spatial_channels: int = 1,
        return_time: bool = False,
        **_ignored,
    ):
        self.data_dir = data_dir
        self.subset = subset
        self.use_sync_encoder = use_sync_encoder

        self.audiovisual = audiovisual
        self.format = format
        self.spatial_channels = int(spatial_channels)
        self.return_time = return_time
        self.normalize_audio = normalize_audio

        self.dummy = dummy
        self.num_frames = int(num_frames)
        self.shuffle_spec = bool(shuffle_spec)
        self.spec_transform = spec_transform or (lambda x: x)

        if stft_kwargs is None:
            raise ValueError(
                "stft_kwargs must include: n_fft, hop_length, win_length, window, center, return_complex"
            )
        required = ["n_fft", "hop_length", "win_length", "window", "center", "return_complex"]
        assert all(k in stft_kwargs for k in required), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        assert bool(self.stft_kwargs.get("center", True)) is True, "'center' must be True"

        # --- Resolve data lists ---
        if format == "custom_data":
            self.data_path = data_dir
            self.clean_audio_path = join(data_dir, "clean_audio")
            self.noisy_audio_path = join(data_dir, "noisy_audio")
            self.lip_features_path = join(data_dir, "lip_features")

            # Print folders
            print(f"找到数据文件夹: {self.clean_audio_path}")
            print(f"找到数据文件夹: {self.noisy_audio_path}")
            print(f"找到数据文件夹: {self.lip_features_path}")

            # Lips stats (directories and .pt files)
            if os.path.isdir(self.lip_features_path):
                lip_dirs = [d for d in glob(join(self.lip_features_path, "*")) if os.path.isdir(d)]
                print(f"找到唇动目录: {len(lip_dirs)} 个")
                lip_files = glob(join(self.lip_features_path, "*/*.pt"))
                print(f"找到唇动帧文件: {len(lip_files)} 个")
            else:
                print(f"警告: 唇部特征路径不存在: {self.lip_features_path}")
                # 仍然允许训练，但会用零占位视觉

            self.clean_files = sorted(glob(join(self.clean_audio_path, "*.wav")))
            self.noisy_files = sorted(glob(join(self.noisy_audio_path, "*.wav")))

            print(f"找到清洁音频文件: {len(self.clean_files)} 个")
            print(f"找到噪声音频文件: {len(self.noisy_files)} 个")

            self.sample_num = min(len(self.clean_files), len(self.noisy_files)) if self.clean_files and self.noisy_files else 0
            print(f"设置样本数量: {self.sample_num}")

            self.sample_rate = 16000
        else:
            self.clean_files, self.noisy_files = [], []
            self.sample_num = 0
            self.sample_rate = 16000

    def __len__(self):
        return self.sample_num

    # --------- visual helpers ---------
    @staticmethod
    def _to_image_tensor_112x112(t: torch.Tensor) -> torch.Tensor | None:
        """Convert an arbitrary frame tensor to [112,112] grayscale in 0..1."""
        t = torch.as_tensor(t).float().cpu()
        if t.ndim == 2:
            img = t
        elif t.ndim == 3:
            if t.shape[0] in (1, 3):  # C,H,W
                if t.shape[0] == 3:
                    img = 0.2989 * t[0] + 0.5870 * t[1] + 0.1140 * t[2]
                else:
                    img = t[0]
            elif t.shape[-1] in (1, 3):  # H,W,C
                if t.shape[-1] == 3:
                    img = 0.2989 * t[..., 0] + 0.5870 * t[..., 1] + 0.1140 * t[..., 2]
                else:
                    img = t[..., 0]
            else:
                return None
        elif t.ndim == 1:
            L = t.numel()
            s = int(np.sqrt(L))
            if s * s == L:
                img = t.view(s, s)
            else:
                return None
        else:
            return None

        if img.numel() == 0:
            return None
        mn, mx = float(img.min()), float(img.max())
        if mx > mn:
            img = (img - mn) / (mx - mn)
        else:
            img = img * 0
        if img.shape != (112, 112):
            img = torch.nn.functional.interpolate(img[None, None, ...], size=(112, 112), mode="nearest").squeeze(0).squeeze(0)
        return img

    def _load_visual_series(self, base_name: str, T_target: int) -> torch.Tensor:
        """Load per-frame visual series and align to T_target -> [T,112,112].
        Search order: .pt, .npy, then .jpg/.png.
        Missing or unreadable -> zeros[T,112,112].
        """
        vdir = os.path.join(self.lip_features_path, base_name)
        if not os.path.isdir(vdir):
            return torch.zeros((T_target, 112, 112), dtype=torch.float32)

        pt_paths = sorted(glob(join(vdir, "*.pt")))
        npy_paths = sorted(glob(join(vdir, "*.npy")))
        img_paths = sorted(glob(join(vdir, "*.jpg")) + glob(join(vdir, "*.png")))

        frames = []
        if pt_paths:
            for p in pt_paths:
                try:
                    t = torch.load(p, map_location="cpu")
                    if isinstance(t, dict) and "feat" in t:
                        t = t["feat"]
                    img = self._to_image_tensor_112x112(t)
                    if img is not None:
                        frames.append(img)
                except Exception:
                    continue
        elif npy_paths:
            for p in npy_paths:
                try:
                    arr = np.load(p)
                    img = self._to_image_tensor_112x112(torch.from_numpy(np.asarray(arr)))
                    if img is not None:
                        frames.append(img)
                except Exception:
                    continue
        elif img_paths:
            try:
                from PIL import Image
            except Exception:
                frames = []
            else:
                for p in img_paths:
                    try:
                        im = Image.open(p).convert("L")
                        t = torch.from_numpy(np.array(im)).float()
                        img = self._to_image_tensor_112x112(t)
                        if img is not None:
                            frames.append(img)
                    except Exception:
                        continue

        if len(frames) == 0:
            return torch.zeros((T_target, 112, 112), dtype=torch.float32)

        V = torch.stack(frames, dim=0)  # [T_v,112,112]
        Tv = V.shape[0]
        if T_target <= 0:
            return V
        if Tv > T_target:
            start = np.random.randint(0, Tv - T_target + 1) if self.shuffle_spec else (Tv - T_target) // 2
            V = V[start : start + T_target]
        elif Tv < T_target:
            pad = T_target - Tv
            V = torch.cat([V, V.new_zeros(pad, 112, 112)], dim=0)
        return V

    # --------- main fetch ---------
    def __getitem__(self, i: int):
        if self.format in ["voxceleb2_SS", "voxceleb2_ssrd", "voxceleb2_SE"]:
            raise NotImplementedError("VoxCeleb formats are not implemented in this minimal dataset.")

        # 1) Load audio
        x_np, _ = librosa.load(self.clean_files[i], sr=self.sample_rate)
        y_np, _ = librosa.load(self.noisy_files[i], sr=self.sample_rate)
        x = torch.tensor(x_np).unsqueeze(0)  # (1, T)
        y = torch.tensor(y_np).unsqueeze(0)  # (1, T)

        # Align length & channels
        min_len = min(x.size(-1), y.size(-1))
        x, y = x[..., :min_len], y[..., :min_len]
        if x.ndimension() == 2 and self.spatial_channels == 1:
            x, y = x[0].unsqueeze(0), y[0].unsqueeze(0)
        assert self.spatial_channels <= x.size(0), (
            f"You asked too many channels ({self.spatial_channels}) for the given dataset ({x.size(0)})"
        )
        x, y = x[: self.spatial_channels], y[: self.spatial_channels]

        if self.normalize_audio:
            normfac = max(y.abs().max().item(), 1e-8)
            x = x / normfac
            y = y / normfac

        # 2) STFT -> complex spec: (..., F, T)
        X = torch.stft(x, **self.stft_kwargs)
        Y = torch.stft(y, **self.stft_kwargs)
        X, Y = self.spec_transform(X), self.spec_transform(Y)

        # 3) Crop frequency bins to 256 (drop Nyquist) for UNet alignment
        if X.shape[-2] == 257:
            X = X[..., :256, :]
        if Y.shape[-2] == 257:
            Y = Y[..., :256, :]

        # 4) Fix time frames to num_frames across the batch
        T_target = int(self.num_frames) if getattr(self, "num_frames", 0) else 0

        def _fix_T(spec: torch.Tensor, T_target: int, do_shuffle: bool) -> torch.Tensor:
            if T_target <= 0:
                return spec
            T = spec.shape[-1]
            if T == T_target:
                return spec
            if T > T_target:
                start = np.random.randint(0, T - T_target + 1) if do_shuffle else (T - T_target) // 2
                return spec[..., start : start + T_target]
            else:
                pad = T_target - T
                zeros = spec.new_zeros(*spec.shape[:-1], pad)
                return torch.cat([spec, zeros], dim=-1)

        X = _fix_T(X, T_target, self.shuffle_spec)
        Y = _fix_T(Y, T_target, self.shuffle_spec)

        # 5) Load per-frame visual and align to T_target -> [T,112,112]
        base = os.path.splitext(os.path.basename(self.clean_files[i]))[0]
        V = self._load_visual_series(base, T_target) if self.audiovisual else torch.zeros((T_target, 112, 112))

        # Always return a 3‑tuple to satisfy model._step(x,y,visual)
        return X, Y, V


# -------------------------------
# Lightning DataModule
# -------------------------------
class SpecsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        base_dir: str,
        *,
        batch_size: int = 4,
        num_workers: int = 0,
        format: str = "custom_data",
        audiovisual: bool = False,
        num_frames: int = 256,
        shuffle_spec: bool = False,
        spatial_channels: int = 1,
        return_time: bool = False,
        # STFT params
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 512,
        window: str = "hann",
        center: bool = True,
        use_sync_encoder: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.base_dir = base_dir
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.format = format
        self.audiovisual = audiovisual
        self.num_frames = int(num_frames)
        self.shuffle_spec = shuffle_spec
        self.spatial_channels = int(spatial_channels)
        self.return_time = return_time
        self.use_sync_encoder = use_sync_encoder

        self.spec_transform = kwargs.get("spec_transform", lambda z: z)

        self.stft_kwargs = dict(
            n_fft=int(n_fft),
            hop_length=int(hop_length),
            win_length=int(win_length),
            window=get_window(window, int(win_length)),
            center=bool(center),
            return_complex=True,
        )

        self.train_set = None
        self.valid_set = None

    @staticmethod
    def add_argparse_args(group):
        group.add_argument("--dm-base-dir", dest="base_dir", type=str, default="my_test_data/train",
                           help="Base directory (expects clean_audio/, noisy_audio/, optional lip_features/).")
        group.add_argument("--dm-batch-size", dest="batch_size", type=int, default=4)
        group.add_argument("--dm-num-workers", dest="num_workers", type=int, default=0)

        group.add_argument("--dm-format", dest="format", type=str, default="custom_data")
        group.add_argument("--dm-audiovisual", dest="audiovisual", action="store_true")
        group.add_argument("--dm-num-frames", dest="num_frames", type=int, default=256)
        group.add_argument("--dm-shuffle-spec", dest="shuffle_spec", action="store_true")
        group.add_argument("--dm-spatial-channels", dest="spatial_channels", type=int, default=1)
        group.add_argument("--dm-return-time", dest="return_time", action="store_true")
        group.add_argument("--dm-use-sync-encoder", dest="use_sync_encoder", action="store_true")

        group.add_argument("--dm-n-fft", dest="n_fft", type=int, default=512)
        group.add_argument("--dm-hop-length", dest="hop_length", type=int, default=160)
        group.add_argument("--dm-win-length", dest="win_length", type=int, default=512)
        group.add_argument("--dm-window", dest="window", type=str, default="hann")
        group.add_argument("--dm-center", dest="center", action="store_true")
        group.add_argument("--dm-no-center", dest="center", action="store_false")
        group.set_defaults(center=True)
        return group

    def prepare_data(self):
        pass

    def setup(self, stage: str | None = None):
        common = dict(
            data_dir=self.base_dir,
            subset="train",
            dummy=False,
            shuffle_spec=self.shuffle_spec,
            num_frames=self.num_frames,
            format=self.format,
            use_sync_encoder=self.use_sync_encoder,
            audiovisual=self.audiovisual,
            normalize_audio=False,
            spec_transform=self.spec_transform,
            stft_kwargs=self.stft_kwargs,
            spatial_channels=self.spatial_channels,
            return_time=self.return_time,
        )
        self.train_set = Specs(**common)
        self.valid_set = Specs(**common)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )
