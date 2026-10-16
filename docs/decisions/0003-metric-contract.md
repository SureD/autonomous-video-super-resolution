# ADR 0003: Quality metric contract

Use RGB uint8 images in [0,255], rounded/clipped once after inference. Y is
BT.709 full range: 0.2126 R + 0.7152 G + 0.0722 B, computed in float64.
No limited-range conversion or 16-level offset is applied. PSNR uses peak 255.
No explicit border crop is allowed. SSIM uses an 11x11 Gaussian (sigma 1.5),
C1=(0.01*255)^2, C2=(0.03*255)^2, and only valid window centers (therefore
five pixels at each edge cannot be SSIM centers); images must be at least 11x11.

Match (relative parent directory as clip_id, filename stem as frame_id), rejecting
extension aliases with duplicate identities. Flat folders are one clip named '.'.
Folder comparisons never crop GT. Generated LR may use only explicitly requested
scale modcrop, recorded in output. Default rejects dimensions indivisible by scale.
Compute spatial metrics per frame, arithmetic means per clip, then equal-weight
clip means and population standard deviations. Temporal pairs stay within clips;
lexicographic frame order is used, so frame IDs must be consistently zero-padded.
Report frame results and clip means. Temporal metrics are diagnostics, without
occlusion masking or a claim of exact TecoGAN metric equivalence.

Scores from earlier BT.601 limited-range/cropped runs are invalid for comparison.
Published scores using other conventions are not directly comparable.
