#!/usr/bin/env python3
"""Build HR / LR frame folders (+ optional H.264 / AV1 re-encoded LR) from a video or a frame folder.

Output layout (per clip, under --out/<name>/):
  HR/00001.png ...                  ground-truth frames (RGB, cropped so H,W are multiples of 2*scale)
  LR_x{s}/00001.png ...             bicubic-downsampled LR (clean)
  LR_x{s}_h264_crf{C}/ clip.mp4 + 00001.png ...   LR -> libx264 (yuv420p, BT.709 tags) -> decoded back
  LR_x{s}_av1_crf{C}/  clip.mp4 + 00001.png ...   LR -> libsvtav1 -> decoded back
  meta.json                         everything needed to reproduce + per-folder frame-hash digests
and appends one line to --out/manifest.jsonl.

Dependencies: ffmpeg/ffprobe on PATH, numpy, Pillow.

Clean LR uses the shared Pillow antialiased bicubic protocol (ADR 0002).

Examples
  python scripts/prepare_data.py --src data/raw/xiph_rush_hour_1080p25_f0-47.y4m --name rush_hour \
      --scale 2 --h264-crf 18,23,28,33 --av1-crf 35,45
  python scripts/prepare_data.py --src data/vid4/calendar/original --name vid4_calendar --scale 2
  python scripts/prepare_data.py --src big.mp4 --name x --ss 60 --frames 120 --hr-height 1080 --scale 2
  # 1080p -> 4K task: the HR is 2160p, LR 1080p
  python scripts/prepare_data.py --src clip4k.mkv --name x4k --hr-height 2160 --scale 2
"""
import argparse, datetime, glob, hashlib, json, os, shutil, subprocess, sys
from concurrent.futures import ProcessPoolExecutor

import _path  # noqa
from vsr.resize import bicubic_down, KERNEL
import numpy as np
from PIL import Image

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")
TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        raise RuntimeError("command failed: " + " ".join(cmd) + "\n" + r.stderr[-2000:])
    return r


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def folder_digest(d):
    """sha256 over the sorted (name, sha256(content)) list of all png files: order/content sensitive."""
    h = hashlib.sha256()
    files = sorted(f for f in os.listdir(d) if f.lower().endswith(".png"))
    for f in files:
        h.update(f.encode()); h.update(sha256_file(os.path.join(d, f)).encode())
    return {"n": len(files), "sha256": h.hexdigest()}


def probe(path):
    r = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,r_frame_rate,pix_fmt,codec_name", "-of", "json", path])
    s = json.loads(r.stdout)["streams"][0]
    n, d = s["r_frame_rate"].split("/")
    s["fps"] = float(n) / float(d)
    return s


def ffmpeg_version():
    return run(["ffmpeg", "-version"]).stdout.splitlines()[0]


# ---------- HR frames ----------
def extract_hr(src, hr_dir, a, w, h):
    """Return (n_frames, fps). Video: ffmpeg decode -> RGB png. Folder: copy/crop via PIL."""
    os.makedirs(hr_dir, exist_ok=True)
    mult = 2 * a.scale
    if os.path.isdir(src):
        files = sorted(f for f in glob.glob(os.path.join(src, "*")) if f.lower().endswith(IMG_EXT))
        files = files[a.start_frame:][: a.frames or None]
        for i, f in enumerate(files, 1):
            im = Image.open(f).convert("RGB")
            W, H = im.size
            if a.hr_height and H != a.hr_height:
                W2 = round(W * a.hr_height / H); im = im.resize((W2, a.hr_height), Image.LANCZOS); W, H = im.size
            im.crop((0, 0, W - W % mult, H - H % mult)).save(os.path.join(hr_dir, f"{i:05d}.png"))
        return len(files), a.fps or 24.0
    info = probe(src)
    vf = []
    if a.hr_height and info["height"] != a.hr_height:
        vf.append(f"scale=-2:{a.hr_height}:flags=lanczos")
    # YUV->RGB with the right matrix (swscale would otherwise assume BT.601 for untagged input)
    vf.append("scale=in_color_matrix=bt709:in_range=tv:out_range=pc" if info["height"] >= 720 else
              "scale=in_color_matrix=bt601:in_range=tv:out_range=pc")
    vf.append(f"crop=trunc(iw/{mult})*{mult}:trunc(ih/{mult})*{mult}:0:0")
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if a.ss:
        cmd += ["-ss", str(a.ss)]
    cmd += ["-i", src]
    if a.frames:
        cmd += ["-frames:v", str(a.frames)]
    cmd += ["-vf", ",".join(vf), "-pix_fmt", "rgb24", "-start_number", "1", os.path.join(hr_dir, "%05d.png")]
    run(cmd)
    return len(os.listdir(hr_dir)), a.fps or info["fps"]


# ---------- bicubic downsample ----------
def _down_one(args):
    src, dst, s, method = args
    if method != "pil":
        raise ValueError("only the shared pil downsample protocol is supported")
    with Image.open(src) as im:
        Image.fromarray(bicubic_down(np.array(im.convert("RGB")), s)).save(dst)


def downsample(hr_dir, lr_dir, s, method, jobs):
    os.makedirs(lr_dir, exist_ok=True)
    files = sorted(os.listdir(hr_dir))
    with ProcessPoolExecutor(jobs) as ex:
        list(ex.map(_down_one, [(os.path.join(hr_dir, f), os.path.join(lr_dir, f), s, method) for f in files], chunksize=4))


# ---------- codec round trip ----------
def codec_roundtrip(lr_dir, out_dir, codec, crf, fps, n_frames):
    os.makedirs(out_dir, exist_ok=True)
    mp4 = os.path.join(out_dir, "clip.mp4")
    if codec == "h264":
        venc = ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p"]
        ver = "libx264 preset=medium"
    elif codec == "av1":
        venc = ["-c:v", "libsvtav1", "-preset", "6", "-crf", str(crf), "-pix_fmt", "yuv420p"]
        ver = "libsvtav1 preset=6"
    else:
        raise ValueError(codec)
    run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", os.path.join(lr_dir, "%05d.png"),
         "-vf", "scale=out_color_matrix=bt709:out_range=tv", *venc, *TAGS, mp4])
    run(["ffmpeg", "-v", "error", "-y", "-i", mp4, "-vf", "scale=in_color_matrix=bt709:in_range=tv:out_range=pc",
         "-pix_fmt", "rgb24", "-start_number", "1", os.path.join(out_dir, "%05d.png")])
    size = os.path.getsize(mp4)
    return {"encoder": ver, "crf": crf, "mp4_bytes": size, "bitrate_kbps": round(size * 8 / (n_frames / fps) / 1000, 1),
            "mp4_sha256": sha256_file(mp4)}


def find_license(src, raw_manifest, name=None):
    """Look up license/provenance for the source in manifest_raw.jsonl: explicit --provenance name, else by file name (best effort)."""
    if not raw_manifest or not os.path.exists(raw_manifest):
        return None
    base = os.path.basename(os.path.normpath(src))
    with open(raw_manifest) as f:
        for line in f:
            r = json.loads(line)
            if (name and r["name"] == name) or (not name and (os.path.basename(r["path"]) == base or (os.path.isdir(src) and r["name"] in src))):
                return {k: r.get(k) for k in ("name", "source_url", "license", "license_evidence", "sha256", "partial")}
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="video file or folder of frames (GT)")
    p.add_argument("--name", required=True)
    p.add_argument("--out", default="data/prepared")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--down", choices=["pil"], default="pil")
    p.add_argument("--hr-height", type=int, default=0, help="resize source to this height first (Lanczos), 0 = keep")
    p.add_argument("--ss", type=float, default=0, help="start time (s) for video input")
    p.add_argument("--start-frame", type=int, default=0, help="first frame for folder input")
    p.add_argument("--frames", type=int, default=0, help="max frames, 0 = all")
    p.add_argument("--fps", type=float, default=0, help="override fps (folder input defaults to 24)")
    p.add_argument("--h264-crf", default="", help="comma list, e.g. 18,23,28,33")
    p.add_argument("--av1-crf", default="", help="comma list, e.g. 35,45")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    p.add_argument("--raw-manifest", default="data/manifests/manifest_raw.jsonl")
    p.add_argument("--provenance", default="", help="name of the record in the raw manifest that describes --src (e.g. bbb1080 when --src is an unzipped file)")
    p.add_argument("--keep-existing", action="store_true", help="do not wipe an existing clip folder")
    a = p.parse_args()
    for t in ("ffmpeg", "ffprobe"):
        if shutil.which(t) is None:
            sys.exit(f"{t} not found on PATH")

    root = os.path.join(a.out, a.name)
    if os.path.exists(root) and not a.keep_existing:
        shutil.rmtree(root)
    hr = os.path.join(root, "HR")
    n, fps = extract_hr(a.src, hr, a, 0, 0)
    first = Image.open(os.path.join(hr, "00001.png"))
    HW = first.size[::-1]
    s = a.scale
    lr = os.path.join(root, f"LR_x{s}")
    downsample(hr, lr, s, a.down, a.jobs)
    meta = {
        "name": a.name, "created": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds"),
        "src": a.src, "src_sha256": sha256_file(a.src) if os.path.isfile(a.src) else None,
        "provenance": find_license(a.src, a.raw_manifest, a.provenance or None),
        "frames": n, "fps": fps, "scale": s, "hr_hw": list(HW), "lr_hw": [HW[0] // s, HW[1] // s],
        "downsample": KERNEL, "ffmpeg": ffmpeg_version(), "args": vars(a),
        "folders": {"HR": folder_digest(hr), f"LR_x{s}": folder_digest(lr)}, "codec": {},
    }
    for codec, crfs in (("h264", a.h264_crf), ("av1", a.av1_crf)):
        for c in [x for x in crfs.split(",") if x]:
            d = os.path.join(root, f"LR_x{s}_{codec}_crf{c}")
            info = codec_roundtrip(lr, d, codec, int(c), fps, n)
            meta["codec"][f"{codec}_crf{c}"] = info
            meta["folders"][os.path.basename(d)] = folder_digest(d)
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    mpath = os.path.join(a.out, "manifest.jsonl")
    rec = {k: meta[k] for k in ("name", "created", "src", "src_sha256", "provenance", "frames", "fps",
                                "scale", "hr_hw", "lr_hw", "downsample", "folders", "codec")}
    old = [l for l in (open(mpath).read().splitlines() if os.path.exists(mpath) else []) if json.loads(l)["name"] != a.name]
    with open(mpath, "w") as f:  # one line per clip name; re-running a clip replaces its line
        f.write("".join(l + "\n" for l in old) + json.dumps(rec) + "\n")
    print(f"{a.name}: {n} frames HR {HW[1]}x{HW[0]} -> LR {HW[1]//s}x{HW[0]//s} ({a.down}); "
          f"codecs: {', '.join(meta['codec']) or 'none'}")


if __name__ == "__main__":
    main()
