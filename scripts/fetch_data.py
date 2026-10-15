#!/usr/bin/env python3
"""Fetch the small, clearly-licensed sources listed in docs/datasets.md and record a manifest.

Sources (all verified with HTTP HEAD on 2026-10-15; see docs/datasets.md):
  vid4        people.csail.mit.edu/celiu/CVPR2011/videoSR.zip   (328 MB; no license stated)
  bbb1080     download.blender.org BBB 1080p30 mp4 zip          (275 MB; CC-BY 3.0)
  xiph:*      media.xiph.org/video/derf y4m, only the first/selected N frames via HTTP Range
              (TUM 1080p25: "No restrictions / No Copyright"; Sintel trailer: CC-BY 3.0)

Usage:  python scripts/fetch_data.py --out data/raw [--only vid4,bbb1080,rush_hour,...]
Writes <out>/../manifests/manifest_raw.jsonl (one JSON line per file: path, bytes, sha256,
source_url, license, license_evidence, fetched_at). Never pushes or transfers anything.
"""
import argparse, datetime, hashlib, json, os, subprocess, sys, urllib.request

XIPH = "https://media.xiph.org/video/derf/y4m/"
TUM_LIC = "TUM 1080p set: ReadMe says 'Restrictions of use: No restrictions; Copyright: No Copyright' (https://media.xiph.org/video/derf/ftp.ldv.e-technik.tu-muenchen.de/pub/test_sequences/1080p/ReadMe_1080p.txt)"
SINTEL_LIC = "CC-BY 3.0 (https://durian.blender.org/sharing/); attribution: (c) copyright 2010, Blender Foundation / durian.blender.org"
BBB_LIC = "CC-BY 3.0 (https://peach.blender.org/about/); attribution: (c) copyright 2008, Blender Foundation / www.bigbuckbunny.org"

# name -> spec
SOURCES = {
    "vid4": dict(kind="file", url="https://people.csail.mit.edu/celiu/CVPR2011/videoSR.zip", dest="videoSR.zip",
                 license="NO LICENSE STATED on author page (research benchmark; treat as research-use only)",
                 evidence="https://people.csail.mit.edu/celiu/CVPR2011/default.html"),
    "bbb1080": dict(kind="file", url="https://download.blender.org/demo/movies/BBB/bbb_sunflower_1080p_30fps_normal.mp4.zip",
                    dest="bbb_sunflower_1080p_30fps_normal.mp4.zip", license=BBB_LIC, evidence=BBB_LIC),
}
for n, f in [("rush_hour", 1555203036), ("sunflower", 1555203036), ("tractor", 2146180176), ("pedestrian_area", 1166402286)]:
    SOURCES[n] = dict(kind="y4m", url=f"{XIPH}{n}_1080p25.y4m", dest=f"xiph_{n}_1080p25_f0-47.y4m", start=0, frames=48,
                      license=TUM_LIC, evidence=TUM_LIC)
SOURCES["sintel_trailer"] = dict(kind="y4m", url=f"{XIPH}sintel_trailer_2k_1080p24.y4m",
                                 dest="xiph_sintel_trailer_1080p24_f300-347.y4m", start=300, frames=48,
                                 license=SINTEL_LIC, evidence=SINTEL_LIC)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rng(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def fetch_y4m(spec, path):
    head = rng(spec["url"], 0, 255)
    hl = head.index(b"\n") + 1
    header = head[:hl]
    w = int(next(t[1:] for t in header.split() if t.startswith(b"W")))
    h = int(next(t[1:] for t in header.split() if t.startswith(b"H")))
    fs = 6 + w * h * 3 // 2  # "FRAME\n" + 4:2:0 8-bit
    a = hl + spec["start"] * fs
    b = a + spec["frames"] * fs - 1
    with open(path, "wb") as f:
        f.write(header)
        # chunked to limit memory
        pos = a
        while pos <= b:
            e = min(pos + (64 << 20) - 1, b)
            f.write(rng(spec["url"], pos, e)); pos = e + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    names = [n for n in (a.only.split(",") if a.only else SOURCES) if n]
    os.makedirs(a.out, exist_ok=True)
    man = os.path.join(os.path.dirname(os.path.abspath(a.out)), "manifests", "manifest_raw.jsonl")
    os.makedirs(os.path.dirname(man), exist_ok=True)
    for n in names:
        s = SOURCES[n]; p = os.path.join(a.out, s["dest"])
        if not os.path.exists(p):
            print("fetching", n, s["url"], flush=True)
            if s["kind"] == "file":
                subprocess.run(["curl", "-sSfL", "-o", p, s["url"]], check=True)
            else:
                fetch_y4m(s, p)
        rec = dict(name=n, path=p, bytes=os.path.getsize(p), sha256=sha256(p), source_url=s["url"],
                   license=s["license"], license_evidence=s["evidence"],
                   partial={k: s[k] for k in ("start", "frames") if k in s} or None,
                   fetched_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds"))
        with open(man, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("ok", n, rec["bytes"], rec["sha256"][:12])


if __name__ == "__main__":
    sys.exit(main())
