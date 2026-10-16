# ADR 0002: Shared antialiased bicubic degradation

## Context

Training used OpenCV INTER_CUBIC (no antialiasing), while prepared LR used
Pillow bicubic. This changes the task and invalidates clean comparisons.

## Decision

Use `vsr.resize.bicubic_down` everywhere clean frame-folder LR is generated:
training, preparation, and evaluation. Input/output are RGB uint8. Top-left
modcrop removes trailing rows/columns to an integer scale multiple. Pillow
BICUBIC uses Keys cubic (a=-0.5), widening the filter for antialiased reduction,
with Pillow boundary handling and uint8 clipping/rounding. Record Pillow version.
This matches Pillow exactly; it approximates MATLAB-style antialiased imresize,
without claiming MATLAB bit equivalence (boundaries and quantization differ).
Training applies crops/augmentation before degradation; equality is defined for
the same resulting HR pixels. SyntheticSR's float tensor fixture is not the
frame-folder degradation protocol.

Preparation now accepts only `--down pil`; regenerate old OpenCV/ffmpeg LR.
Evaluation uses Pillow bicubic upsampling of the same LR as the model for its
floor. Quality follows ADR 0003 (BT.709 full-range Y, no explicit crop). SR is clipped/rounded to
uint8. Provided LR is used for both model and floor; its provenance is external.

## Consequences

Old kernel results are not directly comparable. Prepared datasets must be
regenerated and versioned. Pillow becomes a training dependency. Different
training crop boundaries can produce different LR near crop edges from a crop
of a precomputed full-frame LR. No temporal quality or real-time claim follows
from this clean spatial baseline.
