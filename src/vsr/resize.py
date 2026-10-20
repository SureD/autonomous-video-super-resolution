"""Canonical RGB uint8 resizing for the clean SR degradation protocol."""
import numpy as np

KERNEL = "pillow-bicubic-antialias-uint8"


def bicubic_down(img, scale):
    """Top-left modcrop then antialiased Pillow bicubic (Keys a=-0.5).

    RGB uint8 HWC in/out; integer scale >= 1. Pillow widens the filter for
    reduction and clips/rounds to uint8. This is PIL-exact, not MATLAB-exact:
    boundary handling and intermediate quantization can differ from imresize.
    """
    if not isinstance(scale, int) or isinstance(scale, bool) or scale < 1:
        raise ValueError("scale must be a positive integer")
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("expected RGB uint8 HWC")
    h, w = img.shape[:2]
    h, w = h - h % scale, w - w % scale
    if not h or not w:
        raise ValueError("frame smaller than scale")
    from PIL import Image
    return np.array(Image.fromarray(np.ascontiguousarray(img[:h, :w])).resize(
        (w // scale, h // scale), Image.Resampling.BICUBIC))


def bicubic_up(img, scale):
    """Pillow bicubic RGB uint8 upsampling, for the evaluation floor."""
    h, w = img.shape[:2]
    from PIL import Image
    return np.array(Image.fromarray(img).resize((w * scale, h * scale), Image.Resampling.BICUBIC))
