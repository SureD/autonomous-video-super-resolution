"""PyTorch FP16 latency/FPS benchmark (CUDA events + warmup), ONNX export, optional TensorRT (trtexec).

Example: python scripts/benchmark.py --models espcn fsrcnn ecbsr plain --out experiments/exp001-latency-baselines/results
Measures network-only latency for 960x540 -> 1920x1080 (x2, batch 1). Optional --e2e adds H2D upload of a uint8
frame, normalisation, D2H download of uint8 output (decode/display not included; documented in exp001 README).
"""
import argparse, datetime, json, os, platform, shutil, subprocess, sys, time
import _path  # noqa
import torch
from vsr.models import build


def export_onnx(model, path, h, w, dynamic=False):
    model = model.float().cpu().eval()
    x = torch.randn(1, 3, h, w)
    kw = dict(input_names=["lr"], output_names=["sr"], opset_version=17)
    if dynamic:
        kw["dynamic_axes"] = {"lr": {2: "h", 3: "w"}, "sr": {2: "H", 3: "W"}}
    try:
        torch.onnx.export(model, x, path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, x, path, **kw)
    return path


@torch.no_grad()
def bench_torch(model, h, w, dtype, warmup, iters, e2e, channels_last=False):
    dev = torch.device("cuda")
    model = model.to(dev, dtype).eval()
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    frame = torch.randint(0, 255, (h, w, 3), dtype=torch.uint8).pin_memory()
    x = torch.rand(1, 3, h, w, device=dev, dtype=dtype)
    def step():
        if e2e:
            g = frame.to(dev, non_blocking=True).permute(2, 0, 1)[None].to(dtype).div_(255)
            y = model(g)
            return y.clamp_(0, 1).mul_(255).round_().to(torch.uint8).permute(0, 2, 3, 1).cpu()
        return model(x)
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        s.record(); step(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    mean = sum(times) / len(times)
    return {"mean_ms": mean, "median_ms": times[len(times) // 2], "p95_ms": times[int(0.95 * len(times)) - 1],
            "fps_mean": 1000.0 / mean}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["espcn", "fsrcnn", "ecbsr", "plain"])
    ap.add_argument("--height", type=int, default=540); ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--warmup", type=int, default=50); ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--e2e", action="store_true", help="include uint8 H2D/normalize/D2H in the timed region")
    ap.add_argument("--out", default="experiments/exp001-latency-baselines/results")
    ap.add_argument("--onnx", action="store_true", help="export ONNX for each model")
    ap.add_argument("--trt", action="store_true", help="run trtexec --fp16 if found (implies --onnx)")
    ap.add_argument("--trtexec", default="trtexec")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    have_cuda = torch.cuda.is_available()
    env = {"time": datetime.datetime.now().astimezone().isoformat(), "host": platform.node(), "os": platform.platform(),
           "python": sys.version.split()[0], "torch": torch.__version__, "cuda_available": have_cuda,
           "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if have_cuda else None,
           "gpu": torch.cuda.get_device_name(0) if have_cuda else None}
    torch.backends.cudnn.benchmark = True
    results = {"env": env, "config": vars(a), "models": {}}
    for name in a.models:
        m = build(name)
        if hasattr(m, "merge"):
            m.merge()
        r = {"params": sum(p.numel() for p in m.parameters())}
        if have_cuda:
            dt = torch.float16 if a.dtype == "fp16" else torch.float32
            r["torch_net"] = bench_torch(m, a.height, a.width, dt, a.warmup, a.iters, e2e=False)
            if a.e2e:
                r["torch_e2e_copy"] = bench_torch(m, a.height, a.width, dt, a.warmup, a.iters, e2e=True)
        else:
            r["torch_net"] = "skipped: no CUDA"
        if a.onnx or a.trt:
            p = export_onnx(m, os.path.join(a.out, f"{name}_x2_{a.width}x{a.height}.onnx"), a.height, a.width)
            r["onnx"] = p
            m = m.float()
            cmd = [a.trtexec, f"--onnx={p}", "--fp16", f"--saveEngine={p[:-5]}_fp16.engine",
                   "--warmUp=500", "--iterations=300", "--useCudaGraph", "--noDataTransfers"]
            r["trtexec_cmd"] = " ".join(cmd)
            if a.trt and shutil.which(a.trtexec):
                t0 = time.time()
                proc = subprocess.run(cmd, capture_output=True, text=True)
                log = os.path.join(a.out, f"{name}_trtexec.log")
                open(log, "w").write(proc.stdout + proc.stderr)
                r["trtexec_log"], r["trtexec_rc"], r["trtexec_wall_s"] = log, proc.returncode, time.time() - t0
                for line in proc.stdout.splitlines():
                    if "Throughput:" in line or "GPU Compute Time:" in line:
                        r.setdefault("trtexec_summary", []).append(line.strip())
            elif a.trt:
                r["trtexec_status"] = "trtexec not found on PATH; run trtexec_cmd manually"
        results["models"][name] = r
        print(name, json.dumps(r, default=str))
    tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = os.path.join(a.out, f"bench_{a.dtype}_{a.width}x{a.height}_{tag}.json")
    json.dump(results, open(out, "w"), indent=2, default=str)
    print("wrote", out)


if __name__ == "__main__":
    main()
