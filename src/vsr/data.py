"""Frame-folder dataset: HR frames -> LR via bicubic, optional H.264 re-encode (ffmpeg)."""
import glob, os, random, shutil, subprocess, tempfile
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from vsr.resize import bicubic_down

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def list_frames(root):
    return sorted(p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True) if p.lower().endswith(IMG_EXT))


def h264_roundtrip(img, crf=23, ffmpeg="ffmpeg"):
    """Encode one RGB uint8 frame with libx264 and decode it back. Requires ffmpeg on PATH."""
    if shutil.which(ffmpeg) is None:
        raise RuntimeError("ffmpeg not found on PATH; disable --h264 or install ffmpeg")
    h, w = img.shape[:2]
    h2, w2 = h - h % 2, w - w % 2
    img = np.ascontiguousarray(img[:h2, :w2])
    with tempfile.TemporaryDirectory() as d:
        mp4 = os.path.join(d, "x.mp4")
        subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w2}x{h2}",
                        "-i", "-", "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", mp4],
                       input=img.tobytes(), check=True)
        out = subprocess.run([ffmpeg, "-loglevel", "error", "-i", mp4, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True, check=True).stdout
    dec = np.frombuffer(out, np.uint8)[: h2 * w2 * 3].reshape(h2, w2, 3)
    res = img.copy() if (h2, w2) == (h, w) else None
    if res is None:
        res = np.zeros((h, w, 3), np.uint8); res[:h2, :w2] = dec; return res
    return dec.copy()


def to_tensor(img):
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0


class FrameFolderSR(Dataset):
    def __init__(self, root, scale=2, patch=96, h264_prob=0.0, crf_range=(18, 35), augment=True, length=None):
        self.files = list_frames(root)
        if not self.files:
            raise FileNotFoundError(f"no frames under {root}")
        self.scale, self.patch, self.h264_prob, self.crf_range, self.augment = scale, patch, h264_prob, crf_range, augment
        self.length = len(self.files)  # epoch_len never truncates the population

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(self.files[i]), cv2.COLOR_BGR2RGB)
        s, p = self.scale, self.patch * self.scale
        h, w = img.shape[:2]
        if self.patch and h >= p and w >= p:
            y, x = random.randint(0, h - p), random.randint(0, w - p)
            img = img[y:y + p, x:x + p]
        else:
            img = img[: h - h % s, : w - w % s]
        if self.augment:
            if random.random() < 0.5: img = img[:, ::-1]
            if random.random() < 0.5: img = img[::-1]
        img = np.ascontiguousarray(img)
        lr = bicubic_down(img, s)
        if self.h264_prob > 0 and random.random() < self.h264_prob:
            lr = h264_roundtrip(lr, random.randint(*self.crf_range))
        return to_tensor(lr), to_tensor(img)


class SyntheticSR(Dataset):
    """Random smooth images for smoke tests (no data required)."""
    def __init__(self, n=64, scale=2, patch=32):
        self.n, self.scale, self.patch = n, scale, patch

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        hr = torch.rand(3, self.patch // 4, self.patch // 4, generator=g)
        hr = torch.nn.functional.interpolate(hr[None], size=(self.patch * self.scale,) * 2, mode="bicubic", align_corners=False)[0].clamp(0, 1)
        lr = torch.nn.functional.interpolate(hr[None], scale_factor=1 / self.scale, mode="bicubic", align_corners=False, antialias=True)[0].clamp(0, 1)
        return lr, hr
