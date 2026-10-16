import cv2
import numpy as np


def rgb_to_y(img):
    """img: HxWx3 uint8/float in [0,255] RGB -> Y (BT.709 full range, 0-255) float64."""
    img = img.astype(np.float64)
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def psnr_y(a, b, border=0):
    if border != 0:
        raise ValueError("ADR 0003 requires no border crop")
    ya, yb = rgb_to_y(a), rgb_to_y(b)
    if border:
        ya, yb = ya[border:-border, border:-border], yb[border:-border, border:-border]
    mse = np.mean((ya - yb) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def ssim_y(a, b, border=0):
    if border != 0:
        raise ValueError("ADR 0003 requires no border crop")
    if min(a.shape[:2]) < 11 or a.shape != b.shape:
        raise ValueError("matching shapes and at least 11 pixels required")
    x, y = rgb_to_y(a), rgb_to_y(b)
    if border:
        x, y = x[border:-border, border:-border], y[border:-border, border:-border]
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = cv2.getGaussianKernel(11, 1.5); w = np.outer(k, k.T)
    f = lambda z: cv2.filter2D(z, -1, w)[5:-5, 5:-5]
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    return float(np.mean(((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))))


def warp_error(prev, cur, flow_ref_prev=None, flow_ref_cur=None):
    """Flow-warp error: Farneback flow cur->prev on reference frames (default: the frames themselves),
    warp prev to cur, mean abs diff in [0,255] over valid pixels."""
    rp = flow_ref_prev if flow_ref_prev is not None else prev
    rc = flow_ref_cur if flow_ref_cur is not None else cur
    g0, g1 = cv2.cvtColor(rc, cv2.COLOR_RGB2GRAY), cv2.cvtColor(rp, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = g0.shape
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    mx, my = (gx + flow[..., 0]).astype(np.float32), (gy + flow[..., 1]).astype(np.float32)
    warped = cv2.remap(prev, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    d = np.abs(warped.astype(np.float64) - cur.astype(np.float64)).mean(-1)
    return float(d[valid].mean()) if valid.any() else float("nan")
