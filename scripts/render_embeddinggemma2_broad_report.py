#!/usr/bin/env python3
"""Render the broad probe report directly from measured JSON; no fitted numbers."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/research'
def read(name): return json.loads((OUT/f'embeddinggemma2-broad-{name}.json').read_text())
NAMES={'eg2_global_':'EG2 global','eg2_pooled':'EG2 pooled 3×3','eg2_raw':'EG2 raw patches','dino_patch':'DINOv2-B patches',
       'random_global_':'Random tower → global','random_pooled':'Random tower pooled','random_raw':'Random tower raw'}
VIEWS={'crop192':'192×192 crop','native':'Native input'}
LEVELS=['bicubic_x2','bicubic_x4','bicubic_x8','h264_crf23','h264_crf33']

def fmt(stat):
    lo,hi=stat['ci95']
    return f"{stat['value']:.3f} [{lo:.3f}, {hi:.3f}]"

def main():
    s=read('summary'); manifest=read('manifest'); metrics=read('metrics')
    stages={name:read(name) for name in ['eg2','random','dino']}
    lookup={(r['view'],r['encoder'],r['group'],r['target']):r for r in s['correlations']}
    delta={(r['view'],r['encoder'],r['group'],r['target']):r for r in s['paired_control_deltas']}
    sense={(r['view'],r['encoder'],r['kind']):r for r in s['sensitivity']}
    # Conservative screen for spending EG2-specific training budget; not a claim of SR efficacy.
    gates={}
    for name in list(NAMES)[:3]:
        checks=[]
        for view in VIEWS:
            d=delta[view,name,'within_level_macro','lpips']['spearman']
            dp=delta[view,name,'within_level_macro','negative_psnr_y']['spearman']
            mo=next(r for r in s['monotonicity'] if r['view']==view and r['encoder']==name and r['subset']=='bicubic')
            checks.append(dict(view=view,lpips_control_superiority=d['ci95'][0]>0,
                               no_significant_psnr_control_deficit=dp['ci95'][1]>=0,
                               blur_order_75pct=mo['strict_bicubic_fraction']>=.75,
                               shift_median_below_x2=sense[view,name,'shift_right1']['median_distance_ratio_to_x2']<1))
        gates[name]=dict(pass_all=all(all(v for k,v in c.items() if k!='view') for c in checks),checks=checks)
    verdict='GO to a small, gated training confirmation' if any(g['pass_all'] for g in gates.values()) else 'NO-GO for spending EG2-specific training budget'
    decision=dict(verdict=verdict,criteria='At least one EG2 arm must pass both views: within-level macro LPIPS Spearman improvement over its random-tower counterpart has a positive paired 95% CI; no statistically clear PSNR correlation deficit; >=75% strict x2<x4<x8 ordering; median one-pixel-shift distance < x2 distance.',arms=gates)
    (OUT/'embeddinggemma2-broad-decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    lines=['# EmbeddingGemma 2: broader CPU go/no-go probe','',f'Run on 2026-10-17. **{verdict}.** This broader result supersedes the provisional two-image screen retained at the end. This is an encoder sensitivity/association screen, not a trained SR or VSR result.','',
      '## What ran','',
      '- **40 distinct source frames from eight clips**, five uniformly spaced frames per clip; **both 192×192 center crops and full native-frame inputs for all 40 frames**. Clips: four Vid4 sequences, Xiph sunflower/rush_hour, BBB t=120 s, and Sintel trailer. No synthetic clip was included.',
      '- Five fidelity degradations: Pillow bicubic down/up ×2/×4/×8 and **native-resolution H.264 CRF 23/33** (libx264 medium, YUV420p). All source HR frames in each contiguous clip were encoded before sampling/cropping; these are inter-frame GOP sequences, not five isolated I-frames. Codec compression is a separate family, not added on top of bicubic downsampling.',
      '- Three sensitivity controls: one-pixel right translation (replicated left edge), mild HR unsharp masking, and the same unsharp mask on ×2. Sharpening is Pillow `UnsharpMask(radius=1, percent=50, threshold=0)`. Native images are degraded/shifted/sharpened before extracting their corresponding center crop.',
      '- EG2 global vector, final 3×3-pooled vision tokens, final raw patch tokens; **DINOv2-base final patch tokens**; plus the same three EG2 streams after replacing **only its vision tower with a random initialization (seed 1729)**. The random global arm retains the pretrained projection/text stack, isolating the tower change; it is not an entirely random multimodal encoder.',
      '- **CPU only, FP32; EG2/random four intra-op threads, DINO two**, evaluation/inference mode, audio disabled, no student training. Each encoder stage made **722 forwards** (80 HR references, 640 candidates, two unchanged repeats); 2,166 encoder forwards in total. All recorded features are finite. The metric stage measured 640 PSNR-Y/LPIPS pairs.',
      '- **LPIPS-Alex v0.1 ran**, package 0.1.4. ImageNet AlexNet trunk and learned LPIPS calibration were both loaded from explicit local files; no random trunk metric. The `pnet_rand=True` constructor only suppresses its download path, after which all convolution weights/biases are replaced with the verified pretrained AlexNet tensors. LPIPS uses complete RGB views in [-1,1] with no resize.',
      '- PSNR-Y in this probe uses the pre-ADR 0003 formula `Y=16+(65.481 R+128.553 G+24.966 B)/255` (BT.601 limited-range coefficients, peak 255, **border=0**, uint8 decoded RGB). ADR 0003 later locked BT.709 full-range Y with the same peak and border. These scores were not recomputed. Correlations use **negative PSNR-Y**, so positive correlation has the same fidelity direction as positive LPIPS correlation. HR identities and sensitivity controls are excluded from fidelity correlation tables.',
      '', 'Artifacts: [manifest and exact ffmpeg commands](embeddinggemma2-broad-manifest.json), [per-frame metrics](embeddinggemma2-broad-metrics.json), [EG2](embeddinggemma2-broad-eg2.json), [random tower](embeddinggemma2-broad-random.json), [DINO](embeddinggemma2-broad-dino.json), [joined raw measurements/statistics](embeddinggemma2-broad-summary.json), [decision](embeddinggemma2-broad-decision.json), [download hashes](embeddinggemma2-broad-downloads.json). [Probe script](../../scripts/probe_embeddinggemma2_broad.py), [report renderer](../../scripts/render_embeddinggemma2_broad_report.py).','',
      'Validation: saved LPIPS values matched the unmodified pretrained public constructor on crop/native blur/codec pairs with downloads disabled (absolute tolerance 1e−7). Rank calculations matched SciPy on tied/untied checks; an isolated identical-arm check returned exactly zero paired deltas/CIs. All 640 joined rows, 280 correlation summaries, 120 paired differences and three complete 80-case model stages passed finite-value/count checks. Image dimensions/pairing and script syntax checks passed.','', '## Geometry and statistical interpretation','',
      '“Native” denotes the complete native-resolution source view and native-resolution PSNR/LPIPS, **not an encoder operating on every native pixel without resizing**. EG2 uses the pinned official processor at B=280: the 192² crop is enlarged to 768² (48² raw patches, 16² pooled tokens); full native frames are resized to their aspect-preserving budget. DINO uses the complete same view, torchvision float bicubic/antialias resizing, ImageNet normalization, and **the same raw spatial token-grid dimensions** as EG2 (14-pixel rather than 16-pixel patches). Thus a square DINO raster is 672². This matches field of view and patch count, **not raster size, parameter count, or FLOPs**; it avoids DINO’s default central-224 crop discarding most of the native frame. Different encoder streams are not calibrated to a common distance scale.','',
      'EG2 global is the attention-masked mean of final 768-dimensional sequence states, including valid boundary tokens. Pooled tokens are the final vision-tower output after 3×3 pooling; raw tokens come from a hook on the final vision encoder layer before pooling. Padding is excluded. Dense distances average cosine distance at corresponding spatial positions; cosine normalization/accumulation uses float64 over FP32 features. DINO uses final post-layernorm patch tokens with CLS excluded. No intermediate layer was selected from the observed results.','',
      'Spearman ρ and Kendall τ-b below have **95% percentile CIs from 2,000 paired cluster-bootstrap replicates over eight clips**, seed 20261017. Each draw samples eight clips with replacement and retains every selected frame/level; duplicate clip draws receive duplicate weight. The same draws are used for model-minus-control deltas. Frames are not treated as independent bootstrap units. Only eight clips and one random initialization limit precision and generalization; CIs are descriptive and not adjusted for multiple comparisons.','',
      '- **Pooled across levels**: correlation over all 200 frame/degradation pairs per view. This mixes content and severity.',
      '- **Across levels within frame**: mean of 40 per-frame correlations, each using the five fidelity levels; the clip bootstrap is applied to those per-frame statistics. This controls frame content but has only five points per correlation.',
      '- **Within a fixed level**: correlation across 40 frames/content, reported separately for every level. **Within-level macro** is the arithmetic mean of those five correlations; each bootstrap replicate recomputes all five before averaging. This is the central check against a severity-only association.',
      '', '## Quality levels and feature distances','',
      'Means across the 40 frames in each view (equal numbers per clip). All exact per-frame values are linked above.','',
      '| View | Degradation | PSNR-Y dB | LPIPS | EG2 global | EG2 pooled | EG2 raw | DINO patch | Random global | Random pooled | Random raw |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    # Surface every actual preprocessing geometry, in addition to per-frame JSON.
    geometries=[]
    seen=set()
    dino_records={(r['id'],r['view']):r for r in stages['dino']['records']}
    for r in stages['eg2']['records']:
        g=r['geometry']; dg=dino_records[r['id'],r['view']]['geometry']
        key=(r['view'],tuple(g['input_wh']),tuple(g['resized_wh']))
        if key not in seen:
            seen.add(key)
            geometries.append(f"| {VIEWS[r['view']]} | {g['input_wh'][0]}×{g['input_wh'][1]} | {g['resized_wh'][0]}×{g['resized_wh'][1]} | {dg['resized_wh'][0]}×{dg['resized_wh'][1]} | {g['valid_raw_tokens']} | {g['pooled_tokens']} |")
    # Insert geometry before the statistical interpretation paragraph.
    at=next(i for i,x in enumerate(lines) if x.startswith('Spearman ρ and Kendall'))
    lines[at:at]=['| View | Source pixels | EG2 raster | DINO raster | Raw tokens (both) | EG2 pooled |',
                  '|---|---:|---:|---:|---:|---:|',*geometries,'']
    for r in s['aggregate_metrics']:
        lines.append(f"| {VIEWS[r['view']]} | {r['level']} | {r['mean_psnr_y_db']:.3f} | {r['mean_lpips']:.5f} | "+' | '.join(f"{r['mean_distances'][n]:.5f}" for n in NAMES)+' |')
    for group,title in [('all_levels','Pooled across all five fidelity levels'),('within_frame_across_levels','Across fidelity levels within each frame'),('within_level_macro','Across frames/content within a fixed level: macro of five levels')]:
        lines+=['',f'## {title}','','Each cell is estimate [95% CI].','',
                '| View | Encoder | ρ vs LPIPS | τ vs LPIPS | ρ vs −PSNR-Y | τ vs −PSNR-Y |','|---|---|---:|---:|---:|---:|']
        for view in VIEWS:
            for name,label in NAMES.items():
                a=lookup[view,name,group,'lpips'];b=lookup[view,name,group,'negative_psnr_y']
                lines.append(f'| {VIEWS[view]} | {label} | {fmt(a["spearman"])} | {fmt(a["kendall"])} | {fmt(b["spearman"])} | {fmt(b["kendall"])} |')
    lines+=['','## Does pretraining beat random initialization?','','Paired **pretrained minus random-tower** correlation deltas. A positive lower CI bound supports superiority for that association; an interval spanning zero is inconclusive. This comparison isolates the EG2 tower, while global controls keep the trained text/projection stack.','',
            '| View | Association | Arm | Δρ LPIPS | Δτ LPIPS | Δρ −PSNR-Y | Δτ −PSNR-Y |','|---|---|---|---:|---:|---:|---:|']
    for view in VIEWS:
        for group in ['all_levels','within_frame_across_levels','within_level_macro']:
            for name in list(NAMES)[:3]:
                a=delta[view,name,group,'lpips'];b=delta[view,name,group,'negative_psnr_y']
                lines.append(f'| {VIEWS[view]} | {group} | {NAMES[name]} | {fmt(a["spearman"])} | {fmt(a["kendall"])} | {fmt(b["spearman"])} | {fmt(b["kendall"])} |')
    lines+=['','## Translation and sharpening sensitivity','','Distance ratios are per-frame distance(control, HR) / distance(×2, HR), then median over frames; the ratio CI resamples clips. A median >1 means the perturbation produces a larger encoder distance than ×2 blur. A one-pixel translation itself can also reduce pixel fidelity: LPIPS/PSNR measurements are reported rather than assuming every shift is harmless. Sharpened ×2 additionally tests whether the distance direction agrees with the measured fidelity direction.','',
            '| View | Encoder | Shift/×2 [95% CI] | Shift > ×2 frames | HR sharpen/×2 [95% CI] | ×2 sharpen/×2 [95% CI] | ×2 sharpening direction agrees LPIPS / −PSNR |','|---|---|---:|---:|---:|---:|---:|']
    for view in VIEWS:
        for name,label in NAMES.items():
            a,b,c=[sense[view,name,k] for k in ['shift_right1','sharpen_hr','sharpen_x2']]
            def ratio(r): return f"{r['median_distance_ratio_to_x2']:.3f} [{r['ratio_ci95'][0]:.3f}, {r['ratio_ci95'][1]:.3f}]"
            ag=c['direction_agreement_vs_x2']
            lines.append(f"| {VIEWS[view]} | {label} | {ratio(a)} | {a['fraction_distance_above_x2']*100:.1f}% | {ratio(b)} | {ratio(c)} | {ag['lpips']*100:.1f}% / {ag['negative_psnr_y']*100:.1f}% |")
    lines+=['','## Bicubic monotonicity','','Within each frame: strictly increasing distance for ×2 → ×4 → ×8. Median per-frame ρ is against negative PSNR-Y. Codec levels are included in the five-level tables above, ranked by their actual fidelity rather than an assumed cross-family severity order.','',
            '| View | Encoder | Strictly ordered frames | Median per-frame ρ [95% CI] |','|---|---|---:|---:|']
    for r in s['monotonicity']:
        if r['subset']=='bicubic':
            lines.append(f"| {VIEWS[r['view']]} | {NAMES[r['encoder']]} | {r['strict_bicubic_fraction']*100:.1f}% | {r['median_per_frame_spearman']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] |")
    findings=[
        'The earlier blur-only pass is insufficient: every random-tower stream also strictly orders ×2/×4/×8 in 100% of frames in both views. EG2 dense and DINO do the same; EG2 global orders 97.5% of crops and 100% of native inputs.',
        'Pretraining has a measurable benefit in some summaries: all three EG2 crop streams improve the pooled-across-level LPIPS Spearman correlation over their random-tower counterparts (positive paired CI lower bounds). Global EG2 also improves the mean within-frame five-level LPIPS ordering in both views. That does not establish robust fidelity discrimination across content at a fixed degradation.',
    ]
    for view in VIEWS:
        parts=[]
        for name in ['eg2_pooled','eg2_raw']:
            d=delta[view,name,'within_level_macro','lpips']['spearman']
            dp=delta[view,name,'within_level_macro','negative_psnr_y']['spearman']
            parts.append(f"{NAMES[name]} LPIPS Δρ {fmt(d)}, −PSNR-Y Δρ {fmt(dp)}")
        findings.append(f"**{VIEWS[view]}:** "+'; '.join(parts)+'. Thus the dense within-level LPIPS advantage is inconclusive, while its PSNR-ranking deficit is clear for these sampled clips. Global within-level LPIPS superiority is also unsupported.')
    findings += [
        'The learned encoders are less shift-sensitive than the random tower, but native-input translation remains material: one-pixel shift exceeds the ×2 distance in **37.5% of frames for EG2 pooled and 50% for EG2 raw** (DINO: 45%; EG2 global: 15%). Native median shift/×2 ratios are 0.807, 0.904, 0.751 and 0.462 respectively. In contrast, native LPIPS shift/×2 is 0.213; PSNR-Y penalizes that shift more than ×2 by a median 5.506 dB. This is a metric-dependent tradeoff, not proof that all translation sensitivity is wrong.',
        'Mild ×2 sharpening improves measured fidelity: median PSNR-Y rises **0.531 dB on crops / 0.479 dB on native frames**, and median LPIPS ratios are **0.840 / 0.817**. EG2 pooled/raw distances decrease in 90%/95% of crops and 95%/100% of native frames, so that control is mostly sensible. Its median distance reductions are smaller: ratios 0.976/0.984 on crops and 0.927/0.940 on native inputs. This positive sensitivity result does not resolve the fixed-level control deficits.',
        '**Budget recommendation:** do not allocate an EG2 cache/training sweep from these results. Retain DINO as the planned baseline, with its own training/gradient confirmation still required; these data do not statistically establish that DINO is superior to EG2. Revisiting EG2 would require fresh, training-only clip replication and a predeclared objective resolving pixel fidelity versus perceptual tolerance, rather than selecting a favorable summary from this probe.',
    ]
    lines+=['','## Decision and limits','',f'**{verdict}.**','',*sum(([x,''] for x in findings),[]),decision['criteria'], '',
            'This is a conservative screening rule for whether to allocate EG2 training effort, not a validated law connecting rank correlation to SR gains. Neither a positive correlation nor a DINO point estimate proves that its loss has useful reconstruction gradients. Statistical significance of DINO-versus-EG2 differences was not tested. Same-level correlation across frames is still susceptible to clip/content structure; the bootstrap reflects the eight sampled clips and does not remove that confounding.','',
            '| Arm | View | Positive LPIPS Δρ CI | No clear PSNR Δρ deficit | ≥75% blur ordering | Median shift < ×2 |','|---|---|---|---|---|---|']
    for name,g in gates.items():
        for c in g['checks']:
            lines.append(f"| {NAMES[name]} | {VIEWS[c['view']]} | "+' | '.join('yes' if c[k] else 'no' for k in ['lpips_control_superiority','no_significant_psnr_control_deficit','blur_order_75pct','shift_median_below_x2'])+' |')
    lines+=['', '**Did not run:** student/teacher SR training, learned SR outputs, GPU execution or VRAM profiling, encoder backward/gradient usefulness, temporal fidelity/flow metrics, full native-resolution transformer grids, EG2 B=1120 or intermediate-layer sweep, multiple random seeds, DINO random-init control, SigLIP/CLIP/VGG arms, and external held-out clip replication. The native-input encoder results cannot establish preservation of detail removed by their internal resizing. Existing exp001/teacher/deployment gates still apply before any training decision.','',
            '## Runtime, reproducibility and exact sample IDs','',f"DINO and random-tower timings overlap; stage durations must not be summed as end-to-end wall time. Preparation: {manifest['prepare_seconds']:.2f} s. PSNR/LPIPS pair evaluation: {sum(r['seconds'] for r in metrics['records']):.2f} s."]
    lines.append('')
    for stage,r in stages.items():
        repeats=[v for x in r['records'] for v in x.get('unchanged_repeat',{}).values()]
        lines.append(f"- {stage}: load {r['load_seconds']:.2f} s; inference/image-processing and record construction {sum(x['seconds'] for x in r['records']):.2f} s; total stage {r['total_stage_seconds']:.2f} s. Largest unchanged-repeat distance {max(repeats):.3g}.")
    lines+=['',f"Environment: `{json.dumps(stages['eg2']['versions'],sort_keys=True)}`. DINO revision `{read('downloads')['dinov2_revision']}` ([checkpoint](https://huggingface.co/facebook/dinov2-base)); LPIPS implementation [authors' repository](https://github.com/richzhang/PerceptualSimilarity). EG2 revision `{MODEL_REVISION}` is retained from the original probe; SHA256 `{stages['eg2']['model']['sha256']}`. DINO and LPIPS weight hashes are in the linked manifests. Network acquisition occurred only before inference; model stages set `CUDA_VISIBLE_DEVICES=''`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and use local-files-only model loads.",'', '| Clip | Sampled frame names | Source dimensions |','|---|---|---|']
    for clip in manifest['clips']:
        samples=[x for x in manifest['samples'] if x['clip']==clip['clip']]
        lines.append(f"| {clip['clip']} | {', '.join(x['frame'] for x in samples)} | {samples[0]['native_wh'][0]}×{samples[0]['native_wh'][1]} |")
    lines+=['','With the local model/weight paths in the scripts available:','','```bash','for stage in prepare metrics eg2 random; do','  /tmp/eg2-probe-venv/bin/python scripts/probe_embeddinggemma2_broad.py --stage "$stage" --threads 4 --bootstrap 2000','done','/tmp/eg2-probe-venv/bin/python scripts/probe_embeddinggemma2_broad.py --stage dino --threads 2','/tmp/eg2-probe-venv/bin/python scripts/probe_embeddinggemma2_broad.py --stage summarize --bootstrap 2000','/tmp/eg2-probe-venv/bin/python scripts/render_embeddinggemma2_broad_report.py','```','',
            'Model downloads were explicitly staged from the pinned public Hugging Face DINO revision and the torchvision AlexNet URL recorded in [downloads](embeddinggemma2-broad-downloads.json); EG2 reused the previously SHA256-checked local checkpoint. Installing `lpips==0.1.4 scipy==1.18.1` in the previous probe venv supplied the extra metric/statistics packages. Completed per-view records are checkpointed; to rerun a changed protocol, use a fresh output/cache directory instead of mixing resumed results.','',
            '<details>','<summary>Full per-level content correlations, including confidence intervals</summary>','']
    for view in VIEWS:
        for group in LEVELS+['bicubic_only','codec_only']:
            lines += [f'### {VIEWS[view]} / {group}','','| Encoder | ρ LPIPS | τ LPIPS | ρ −PSNR-Y | τ −PSNR-Y |','|---|---:|---:|---:|---:|']
            for name,label in NAMES.items():
                a=lookup[view,name,group,'lpips'];b=lookup[view,name,group,'negative_psnr_y']
                lines.append(f'| {label} | {fmt(a["spearman"])} | {fmt(a["kendall"])} | {fmt(b["spearman"])} | {fmt(b["kendall"])} |')
            lines.append('')
    lines+=['</details>','']
    report=OUT/'embeddinggemma2-probe-report.md'
    old=report.read_text()
    marker='<details>\n<summary>Historical two-frame CPU screen (superseded)</summary>\n\n'
    if marker in old:
        historical=old.split(marker,1)[1].rsplit('\n</details>',1)[0]
    else:
        historical=old
    lines += [marker+historical+'\n</details>','']
    report.write_text('\n'.join(lines))
    print(verdict)

# Same pin used in the original executable probe, imported without heavy dependencies.
from probe_embeddinggemma2 import MODEL_REVISION
if __name__=='__main__': main()
