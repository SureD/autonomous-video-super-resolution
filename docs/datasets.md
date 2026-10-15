# Datasets plan: training / evaluation for lightweight real-time VSR

- Date of checks: **2026-10-15 (Asia/Shanghai)**. All sizes/licences below were read from the cited page or HTTP header during this session unless marked **unverified**.
- Scope: what to use for ×2 real-time VSR (540p→1080p primary, 1080p→4K stretch), how to get it legally, what is already on the agent box, and what would have to move to `dscs2pc` (RTX 3070 PC). **Nothing was transferred to dscs2pc.**
- Scripts: [`scripts/fetch_data.py`](../scripts/fetch_data.py) (small licensed sources + manifest), [`scripts/prepare_data.py`](../scripts/prepare_data.py) (frames, ×2 LR, H.264/AV1 LR variants + manifest).
- Manifests (committed copies): [`docs/manifests/manifest_raw.jsonl`](manifests/manifest_raw.jsonl) (downloads: sha256, URL, licence, date), [`docs/manifests/manifest_prepared.jsonl`](manifests/manifest_prepared.jsonl) (derived frame folders with per-folder digests).

## 0. Key findings (read this first)

1. **There is almost no clean, openly licensed 1080p video for training.** REDS (CC BY 4.0) is 720p. Vimeo-90K is 448×256 and non-commercial. The 1080p material we can use legally is (a) the TUM 1080p25 set on Xiph ("No restrictions / No Copyright", 7 clips, ≤500 frames each), (b) Blender open movies (CC BY 3.0) – but their public downloads are *lossy H.264* (e.g. BBB 1080p30 mp4 is 2.999 Mbps by `ffprobe`), so "GT" carries codec artifacts.
2. **REDS' own LR is ×4 only** (`val_sharp_bicubic/X4/...` seen in the zip entry names). For ×2 we must generate LR ourselves from `*_sharp` with the same bicubic function in train and eval. This is what `prepare_data.py` does.
3. **Vid4 / UDM10 / SPMCS are ×4 benchmarks at ≤576p/720p**: use them only as sanity checks (own ×2 LR); numbers are *not* comparable with ×4 literature tables.
4. **Licences that would bite**: Vimeo-90K ("non-commercial" per the dataset README as quoted by a mirror – original not re-verified), NTIA clips ("research purposes only", text must travel with derivatives), SVT 2160p/1080p clips (only "developing, testing and presenting technology standards"), Netflix 4096×2160 clips (CC BY-NC-ND 4.0 – "NoDerivatives" is ambiguous for training/downscaling; eval-only at most, ask a lawyer before any release), Vid4/UDM10/SPMCS (no data licence stated). If the user ever wants to *release weights or a product*, restrict training to REDS + TUM + Blender (+ own footage).
5. **Direct download from the box**: REDS (Hugging Face, resumable, HEAD+Range OK), Xiph (Range OK, we pull only the frames we need from the raw y4m), Blender (download.blender.org), Vid4 (author site), SPMCS (Tencent COS link via tinyurl, HEAD 200). **Not reachable**: Vimeo-90K data host `data.csail.mit.edu` answered `403 Forbidden` from the box; UDM10 is on TeraBox/OpenDataLab (not scriptable, **unverified**).

## 1. Dataset table

"Box test" = what I actually ran from the agent box. Sizes in decimal GB unless GiB is stated.

| Dataset | Official source (verified) | Size | Licence / terms | Resolution, content | Role in this project | Box test |
|---|---|---|---|---|---|---|
| **REDS** (train/val/test; NTIRE19/20/21) | https://seungjunnah.github.io/Datasets/reds.html (fetched, 200). Data: https://huggingface.co/datasets/snah/REDS (HF API: `license: cc-by-4.0`, not gated); also Google Drive | `train_sharp.zip` 34,261,573,976 B (34.26 GB); `val_sharp.zip` 4,273,865,704 B; `train_sharp_bicubic.zip` (×4 only) 2,652,551,733 B; `val_sharp_bicubic.zip` 330,490,885 B; `train_blur_comp.zip` (MPEG-compressed blur) 22,775,753,809 B; `val_blur_comp.zip` 2,806,159,829 B (all from `Content-Length` of HF `resolve` URLs) | **CC BY 4.0** (stated on the REDS page) | 240 train clips (000–239) + 30 val clips (240–269 after regroup) – clip counts from BasicSR docs. 1280×720, 100 frames/clip, 24 fps: from the dataset paper, **unverified in this session** | **Train** (all clips except REDS4); **Val/Test**: REDS4 = clips 000, 011, 015, 020 of the original train set (BasicSR docs) – these 4 must be excluded from training. Note HR is 720p, so ×2 gives 360p→720p (scale matches, absolute resolution doesn't) | HEAD ✔ (302→200 with length), Range ✔; **not downloaded** (over the 2 GB cap) |
| **Vimeo-90K septuplet** | http://toflow.csail.mit.edu/ (fetched via WebFetch; `curl` from the box timed out); data `http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip` | page: whole septuplet zip **82 GB**; SR test set 6 GB (×4 LR) | No licence on the project page. The dataset README (quoted by Kaggle mirror, found via search) says "for non-commercial usage only" → treat as **non-commercial research only** | 91,701 septuplets, **448×256** (page) | Optional extra low-res training; low value for 540p→1080p (tiny frames, ×4 LR only in test zip). **Recommend skipping** | `data.csail.mit.edu` → **403 Forbidden** from the box; not downloadable here |
| **Vid4** | Liu & Sun, https://people.csail.mit.edu/celiu/CVPR2011/default.html → `videoSR.zip` | 328,495,021 B (HEAD 200, `Last-Modified 2015-02-10`) | **No licence stated** on the page (research benchmark) | calendar 720×576 (41 f), city 704×576 (34 f), foliage 720×480 (49 f), walk 720×480 (47 f); zip also has `input` (×4 LR 180×144 etc.), `bicubic`, `output` folders. Frame counts/sizes measured from the zip | **Test (sanity only)**; use `original/` as GT, own ×2 LR | **Downloaded** ✔ sha256 in manifest |
| **UDM10** | PFNL repo https://github.com/psychopa4/PFNL (code MIT per GitHub API); data re-uploaded to TeraBox per README; also OpenDataLab page | **unverified** | README (search snippet): datasets "should only be used for study" (copyright) | 10 clips × 32 frames, 720×1272 (OpenDataLab / papers-with-code descriptions; not verified by download) | Test (sanity only), ×4 benchmark | Not scriptable (TeraBox/OpenDataLab) → **not downloaded** |
| **SPMCS** | https://github.com/jiangsutx/SPMC_VideoSR (MIT licence on the repo; data licence not stated) → tinyurl → `research-1256631708.cos.ap-guangzhou.myqcloud.com/shared_files/spmc_videosr_test_set.zip` | 1,106,925,104 B (HEAD 200, `Last-Modified 2019-09-27`) | **No data licence stated** | 30 clips × 31 frames, GT 540×960 with ×2/×3/×4 bicubic LR (README + ICCV'17 paper) | Test (sanity); has ×2 LR | HEAD ✔; **not downloaded** (would exceed the ~2 GB budget together with the other downloads) |
| **Xiph/derf – TUM 1080p25** (`rush_hour`, `sunflower`, `tractor`, `pedestrian_area`, `riverbed`, `station2`, `blue_sky`) | https://media.xiph.org/video/derf/ → `y4m/<name>_1080p25.y4m`; readme https://media.xiph.org/video/derf/ftp.ldv.e-technik.tu-muenchen.de/pub/test_sequences/1080p/ReadMe_1080p.txt | rush_hour & sunflower 1,555,203,036 B (500 f); tractor 2,146,180,176 B; pedestrian_area 1,166,402,286 B (375 f); others see page | ReadMe: **"Restrictions of use: No restrictions. Copyright: No Copyright."** (Sony HDW-F900 HDCam, 4:2:0, BT.709, 2001) | 1920×1080 25 fps, uncompressed y4m 4:2:0 (verified with ffprobe on our pieces) | **Eval (1080p GT)**; the cleanest legal 1080p GT available; only 7 clips | Range ✔; **downloaded 48-frame pieces of 4 clips** |
| **Xiph – Sintel trailer 1080p24** | `y4m/sintel_trailer_2k_1080p24.y4m` (3,897,338,763 B); source Blender Durian | 3.9 GB full | **CC BY 3.0** (https://durian.blender.org/sharing/: "licensed under the Creative Commons Attribution 3.0"; attribution "(c) copyright 2010, Blender Foundation / durian.blender.org") | 1920×1080 24 fps, 1253 frames; rendered CG, converted to y4m by Xiph | Eval (animation, clean render) | Range ✔; **downloaded frames 300–347** |
| **Blender – Big Buck Bunny** | https://download.blender.org/demo/movies/BBB/ ; also `peach/bigbuckbunny_movies/` | `bbb_sunflower_1080p_30fps_normal.mp4.zip` 275,524,128 B; `big_buck_bunny_1080p_h264.mov.zip` 725,106,348 B | **CC BY 3.0** (https://peach.blender.org/about/ ; attribution "(c) copyright 2008, Blender Foundation / www.bigbuckbunny.org") | 1080p30 H.264 **2.999 Mbps** (`ffprobe`), 634.5 s – lossy source | Eval/robustness slice only (compressed GT); *not* used for training (kept disjoint) | **Downloaded** the 275 MB zip ✔ |
| **Blender – Tears of Steel** | https://download.blender.org/demo/movies/ToS/ | `tears_of_steel_1080p.mov.zip` 583,774,281 B; `ToS-4k-1920.mov.zip` 738,876,511 B (what "4k-1920" means: **unverified**); `tearsofsteel_4k.mov.zip` 6,737,593,114 B | Film: **CC BY 3.0** (https://mango.blender.org/sharing/). The `copyright.txt` in that folder is CC BY-ND 3.0 but says it applies **only to the soundtrack** | 1080p / 4K live action + VFX, lossy mov | **Train (1080p supplement)** later; 4K candidate for 1080p→4K eval | listing + sizes ✔; not downloaded |
| **Blender – Sintel** (full) | https://download.blender.org/durian/movies/ | `Sintel.2010.1080p.mkv` 1,180,090,590 B; `Sintel.2010.4k.mkv.zip` 4,506,488,537 B | CC BY 3.0 (see above) | 1080p / 4K | 4K eval candidate (1080p→4K) | listing ✔; not downloaded (4K > cap) |
| Blender – Elephants Dream, Spring, Cosmos Laundromat, Sprite Fright, … | https://download.blender.org/ED/ ; https://studio.blender.org/films/ | ED 1024×576 avi/mov 0.33–0.45 GB | ED licence page `orange.blender.org/license/` returned 404; studio film pages are JS, licence text not extracted → **unverified** (commonly CC BY 2.5/3.0/4.0, check each page) | – | Skip until licences are checked per film | – |
| **Xiph – Twitch game captures** (CSGO, DOTA2, GTAV, MINECRAFT, …) | https://media.xiph.org/video/derf/ (Gaming section) ; https://media.xiph.org/video/derf/twitch/copyright.txt | per-file sizes **unverified** (not listed on the page) | `copyright.txt`: "placed in the public domain. The recordings may contain copyrighted work which can only be used in compliance with fair use." | 16:9 game footage; resolution **unverified** | Candidate for the "game capture" hold-out set in ADR 0001 (eval only, do not redistribute) | Not downloaded |
| **Xiph – NTIA** (`aspen`, `red_kayak`, `snow_mnt`, `speed_bag`, `touchdown_pass`, `west_wind_easy`, `controlled_burn`, `rush_field_cuts`) | https://media.xiph.org/video/derf/ (readme mirrored at `.../vqeg.its.bldrdoc.gov/HDTV/NTIA_source/HDTV_Readme.txt`) | 2.3 GB each (page) | **"research purposes, only"**; redistribution/derivatives allowed only with the NTIA/ITS ownership text kept | 1080p30, 4:2:2 | Optional research-only eval; must carry the readme with derivatives | Not downloaded |
| **Xiph – SVT** (`crowd_run`, `ducks_take_off`, `in_to_tree`, `old_town_cross`, `park_joy`; 2160p50 + 1080p50) | https://media.xiph.org/video/derf/ ; readme PDF mirrored at `.../vqeg.its.bldrdoc.gov/HDTV/SVT_MultiFormat/SVT_MultiFormat_v10.pdf` | 1.5 GB (1080p) / 5.8 GB (2160p) | PDF §2: IP stays with SVT; "may only be used for the purpose of developing, testing and presenting technology standards" → **do not use for ML training**; free distribution but no fee | 2160p50 native 4K (film-scanned) | *Would be* ideal 4K GT technically, but terms are restrictive – only consider for pure testing after the user accepts the terms | Not downloaded |
| **Xiph – Netflix 4096×2160 10-bit** (Boat, Tango, FoodMarket, …) | https://media.xiph.org/video/derf/ → `ElFuente/…y4m`, `Chimera/…y4m` | ~26.5 MB/frame (4096×2160×1.5×2 B, computed), 300 frames ≈ 8 GB (computed) | `ElFuente/Netflix_Boat_Copyright.txt` (fetched): **CC BY-NC-ND 4.0**. The `Chimera/…_Copyright.txt` URLs returned 404 → Chimera clips' terms **unverified** | 4096×2160 59.94 fps 10-bit 4:2:0 | 4K eval candidate (non-commercial research only; ND risk) | Not downloaded |
| **Pexels videos** | https://www.pexels.com/license/ (fetched) | n/a | Free to use and modify, attribution not required; **cannot redistribute** on other stock/wallpaper platforms. Whether *ML training* is permitted: **not stated on that page – unverified**; Pexels API/website terms for bulk download not checked | up to 4K clips, mostly H.264 | Optional manual "consumer content" hold-out (small, hand-picked, never republished) | not tested |
| **Pixabay videos** | https://pixabay.com/service/license-summary/ (fetched) | n/a | Free use, no attribution, modification allowed; no standalone redistribution; "full Content License is the only legally binding" – ML-training clause **unverified** | up to 4K, H.264 | same as Pexels | not tested |
| **Self-made compressed eval set** | built by `prepare_data.py` from the clean 1080p GT above | ≈0.7–1 MB per 960×540 PNG frame per variant (measured) | inherits the source licence | LR 960×540 → H.264 (libx264, preset medium, yuv420p, BT.709 tags) CRF 18/23/28/33 and AV1 (libsvtav1 preset 6) CRF 35/45, decoded back to PNG | **Test (H3 / compression robustness)**; also training-time degradation pool | ✔ see §3 |

## 2. What was downloaded (agent box, `/workspace/data`, 2026-10-15 10:34–10:36 CST)

Total network volume ≈ **1.35 GB** (cap was ~2 GB). Disk: `df -h /workspace` before: 119 GB free of 125 GB; after all prepared outputs (2.7 GB frames) + raw (1.3 GB) + extracted (0.6 GB): 114 GB free.

| Local file (`/workspace/data/raw/…`) | Bytes | sha256 | Source | Licence |
|---|---|---|---|---|
| `videoSR.zip` (Vid4) | 328,495,021 | `ce7087aa787838ec0fdd33f5eab82cdca5d6c7d91bb4cd5cf6807ea3cfda6cc2` | people.csail.mit.edu/celiu/CVPR2011/videoSR.zip | none stated |
| `bbb_sunflower_1080p_30fps_normal.mp4.zip` | 275,524,128 | `e320fef389ec749117d0c1583945039266a40f25483881c2ff0d33207e62b362` | download.blender.org/demo/movies/BBB/ | CC BY 3.0 |
| `xiph_rush_hour_1080p25_f0-47.y4m` | 149,299,524 | `46662f2f6f3f4c8a2422d0ec763164f58c406a5bd7fd5c22b6cfd945073abc67` | media.xiph.org/video/derf/y4m/rush_hour_1080p25.y4m, frames 0–47 via HTTP Range | TUM: no restrictions |
| `xiph_sunflower_1080p25_f0-47.y4m` | 149,299,524 | `cf036dacc74c7107890b25610cefebcc9b9729764176587381d75fc2fa16af45` | same pattern | TUM |
| `xiph_tractor_1080p25_f0-47.y4m` | 149,299,524 | `92f2e9e41a088dd0770ec1ac4b652cacb8f891be77af22737985b1fa81c067ce` | same | TUM |
| `xiph_pedestrian_area_1080p25_f0-47.y4m` | 149,299,524 | `c7fb7b9216eb559015f35f0ee06bf734379709144b5e699edce06b118493a9dc` | same | TUM |
| `xiph_sintel_trailer_1080p24_f300-347.y4m` | 149,299,533 | `c0279f5f465d846b593e101253b0e6310b0343bc639d0d187ea4f6dbb24e7d94` | `sintel_trailer_2k_1080p24.y4m`, frames 300–347 | CC BY 3.0 |

Full hashes, URLs, licence evidence strings and timestamps are in `docs/manifests/manifest_raw.jsonl` (`/workspace/data/manifests/manifest_raw.jsonl` on the box). The `data/` directory is git-ignored; only manifests are committed.

Derived with `scripts/prepare_data.py` (all verified by running it; PIL-bicubic ×2, `/workspace/data/prepared/<clip>/…`): `rush_hour`, `sunflower`, `tractor`, `pedestrian_area`, `sintel_trailer`, `bbb_t120` (BBB from t=120 s, 48 frames) – each 48 frames, HR 1920×1080 → LR 960×540, plus `LR_x2_h264_crf{18,23,28,33}` and `LR_x2_av1_crf{35,45}`; and `vid4_{calendar,city,foliage,walk}` (GT `original/` → ×2 LR + H.264 CRF23). `_synthetic_4k_scripttest` is an ffmpeg `testsrc2` clip used only to prove the 2160p→1080p code path; **it is not a dataset**.

Measured on `rush_hour` (one clip only – do not generalise): LR 960×540@25 fps libx264 bitrate 3036 / 1521 / 804 / 443 kbps at CRF 18 / 23 / 28 / 33; SVT-AV1 880 / 450 kbps at CRF 35 / 45. RGB PSNR of decoded LR vs clean LR (frames 0,8,…,40): 38.41 / 36.86 / 35.06 / 33.04 dB (H.264) and 36.74 / 35.33 dB (AV1). Bicubic ×2 upsample of clean LR vs HR: 42.51 dB RGB PSNR (baseline to beat, same frames).

Disk per 48-frame 1080p clip (measured): HR PNG 81–211 MB (≈1.7–4.4 MB/frame), clean LR PNG 22–39 MB, each codec-variant LR 20–54 MB; whole prepared tree 2.7 GB.

## 3. How to reproduce

```bash
# from repo root; data goes to ./data (git-ignored)
python scripts/fetch_data.py --out data/raw                         # ~1.35 GB total, writes data/manifests/manifest_raw.jsonl
unzip -q data/raw/videoSR.zip -d data/vid4
unzip -q data/raw/bbb_sunflower_1080p_30fps_normal.mp4.zip -d data/bbb
python scripts/prepare_data.py --src data/raw/xiph_rush_hour_1080p25_f0-47.y4m --name rush_hour \
    --scale 2 --h264-crf 18,23,28,33 --av1-crf 35,45                # -> data/prepared/rush_hour/{HR,LR_x2,LR_x2_h264_crf*,LR_x2_av1_crf*,meta.json}
python scripts/prepare_data.py --src data/vid4/calendar/original --name vid4_calendar --fps 25 --scale 2
python scripts/prepare_data.py --src data/bbb/bbb_sunflower_1080p_30fps_normal.mp4 --name bbb_t120 --ss 120 --frames 48 \
    --scale 2 --provenance bbb1080 --h264-crf 23
python scripts/prepare_data.py --src <4k video> --name x --hr-height 2160 --scale 2     # 1080p->4K task
```
`--down pil|cv2|ffmpeg` selects the bicubic implementation (default `pil`; `ffmpeg` path tested; `cv2` path untested because cv2 is not installed on the box). **Caveat:** `src/vsr/data.py` (training) uses `cv2.INTER_CUBIC` (a=−0.75, no antialias), which is *not* identical to PIL's antialiased bicubic. Before exp002 we must choose one function for both training LR synthesis and eval LR, and record it in the ADR. Until then eval folders built with `--down pil` and a model trained with cv2 bicubic would have a train/test degradation mismatch.

## 4. Evaluation protocol sketch (extends ADR 0001; still a proposal)

Common: Y-channel (BT.709 full-range RGB→Y) PSNR/SSIM on the **full frame, no border crop**; LPIPS(alex); tLPIPS; flow-warp error; report per-clip means and the mean±std *over clips* (frames within a clip are not independent). Frame-by-frame bicubic ×2 upsample is the reference baseline (42.51 dB RGB PSNR on `rush_hour` frames 0,8,…,40, see above). Latency is measured separately (exp001).

| ID | Task | GT | LR | Clips (current) | Purpose |
|---|---|---|---|---|---|
| E1 (primary) | 540p→1080p ×2 | 1080p frames (cropped to multiples of 4) | bicubic ÷2 → 960×540 | TUM ×4 + Sintel trailer ×1 (48 frames each; can be extended to 250–500 frames by re-running `fetch_data.py --frames`) | headline PSNR/SSIM/LPIPS/tLPIPS; H1, H2, H4, H5 |
| E2 (stretch) | 1080p→4K ×2 | 2160p frames | bicubic ÷2 → 1920×1080 | **none yet** – candidates: Sintel 4K (CC BY 3.0, 4.5 GB zip), ToS 4K (CC BY 3.0, 0.74/6.7 GB zips, which is which **unverified**), Netflix Boat/… (NC-ND). GT would be lossy-coded; label it as such | 3070 will likely miss 60 FPS here; quality secondary |
| E3 (compression) | 540p→1080p ×2 on coded LR | same HR as E1 | `LR_x2_h264_crf{18,23,28,33}`, `LR_x2_av1_crf{35,45}` | same clips as E1 | H3; also checks clean-trained models on codec artefacts. Report the achieved kbps per CRF |
| E4 (sanity) | ×2 on Vid4 (own LR), REDS4 (own ×2 LR from `*_sharp`, 360p→720p) | native frames | `prepare_data.py` | Vid4 ×4 clips; REDS4 4 clips ×100 frames | compare to similar-scale models trained on REDS; not comparable with ×4 literature numbers |
| E5 (robustness) | consumer content | – | – | BBB slice (compressed GT), Twitch game captures (public domain, fair use caveat), optional hand-picked Pexels/Pixabay | qualitative + tLPIPS; never used for training or tuning |

Leakage rules: REDS4 clips (000, 011, 015, 020) out of training; TUM clips, Sintel (all of it), BBB out of training; training 1080p supplement = Tears of Steel (+ other Blender films after licence check) and REDS. Tune hyper-parameters on REDS val clips (240–269) or a held-out slice of ToS, never on E1.

## 5. Training data plan (proposal)

| Source | Role | Why |
|---|---|---|
| REDS train (240 clips minus REDS4 = 236) | main training HR, 720p (zip is ≈ 34.3 GB, from header) | only large legal clean-ish video set; CC BY 4.0 |
| Tears of Steel 1080p (0.58 GB zip) + extra Blender films | 1080p-native supplement | gives 1080p texture statistics; lossy source – mix with REDS |
| Random-CRF H.264 (and AV1) applied to LR on the fly | degradation for H3 | `src/vsr/data.py --h264`; pre-made variants only needed for eval |
| Vimeo-90K | skip | tiny frames, non-commercial, host blocked from box |

Training from REDS 720p GT only gives ×2 = 360p→720p patches. A 540p→1080p model sees denser LR detail than that; the 1080p supplement is there to reduce this gap, and **whether it matters is an open empirical question** (candidate sub-experiment for exp002).

## 6. What would have to be moved to dscs2pc (not done)

Only on the user's go-ahead; all numbers below are **estimates** from header sizes or measured per-frame sizes above.

| Package | Content | Estimated size | Needed for |
|---|---|---|---|
| A. Eval pack (minimum, Phase 0) | 5 clips × 48 frames: HR 1080p PNG + LR + 4 H.264 + 2 AV1 variants (prepared tree) ≈ 2.1 GB measured (sum of the 5 prepared trees); or ship only raw y4m (5×0.15 GB = 0.75 GB) and rebuild with `prepare_data.py` on the PC | 0.75 GB (raw) / ≈ 2.1 GB (prepared) | E1, E3, exp002 eval |
| B. Vid4 | `videoSR.zip` | 0.33 GB | E4 |
| C. REDS subset | `train_sharp.zip` 34.26 GB unzipped ≈ same (PNG already compressed; **estimate**). Using every 5th frame of the 236 train clips ≈ 6.9 GB; or pre-cropped 96×96 LR / 192×192 HR patches in LMDB (size depends on patch count; ~**unknown**) | 7–34 GB | main training (exp002+) |
| C'. REDS val / REDS4 | `val_sharp.zip` 4.27 GB (30 clips); REDS4 alone ≈ 4/240 of train ≈ 0.57 GB (**estimate** = 34.26 GB × 4/240) | 0.6–4.3 GB | E4, validation |
| D. 1080p supplement | `tears_of_steel_1080p.mov.zip` 0.58 GB (full-film PNG extraction would be tens of GB at ~4 MB/frame measured on other clips – decode on the fly or subsample frames) | 0.6 GB (video) | training supplement |
| E. 4K eval (if wanted) | one 4K source 0.7–6.7 GB zip, cut to ~5 s of frames | 1–7 GB | E2 |
| Total Phase 0 (A raw + B + C every-5th + REDS4 + D) | 0.75 + 0.33 + 6.9 + 0.57 + 0.58 = 9.13 | ≈ 9.1 GB (≈ 10.5 GB if the prepared eval tree is shipped instead of raw) | |

The PC's free disk is **unknown** (not checked – no access allowed). Transfer would be done by the user from the box with `CopyFromBox`-style copy; I have not touched dscs2pc.

## 7. Unverified / not done

- REDS frame resolution (720p), frames per clip (100) and fps – from memory of the paper; clip counts (240/30) and the REDS4 split are confirmed by BasicSR docs.
- Vimeo-90K licence text (only the Kaggle mirror README as quoted in a search result), train/test list sizes, and the data hosts (403 from the box).
- UDM10 size, licence, resolution (second-hand), and any scriptable download.
- SPMCS content (not downloaded), data licence.
- Licences of Elephants Dream / Spring / Cosmos Laundromat / Sprite Fright; Netflix Chimera clips; Xiph hdgreetings clips (`dinner`, `factory`, `life` → `www.hdgreetings.com/copyright.txt` not fetched); Twitch capture sizes/resolution.
- Pexels/Pixabay: explicit permission for ML training; bulk-download terms.
- Whether ToS `ToS-4k-1920.mov.zip` is true 4K (name suggests 1920 wide).
- `prepare_data.py --down cv2` path (no cv2 on the box); AV1 decoding relied on the box's ffmpeg 7.1.5 (libdav1d/libaom availability implicitly OK because decode worked).
