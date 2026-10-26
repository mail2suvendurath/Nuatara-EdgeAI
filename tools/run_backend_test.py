"""
tools/run_backend_test.py — Full integration test: YAML config → backend → infer.

On laptop:
  cd /Users/suvendurath/nuatara-poc/nuatara-percept
  python tools/run_backend_test.py --config configs/laptop.yaml

On Khadas:
  cd /home/khadas/nuatara-percept
  python tools/run_backend_test.py --config configs/khadas-edge2.yaml
"""
import argparse, statistics, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config import EngineConfig
from backends import create_backend

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[1m"; X = "\033[0m"

def ok(label, detail=""): print(f"  {G}✓{X} {label:<22}{('  '+detail) if detail else ''}")
def fail(label, detail=""): print(f"  {R}✗{X} {label:<22}{('  '+detail) if detail else ''}")
def warn(label, detail=""): print(f"  {Y}⚠{X} {label:<22}{('  '+detail) if detail else ''}")

def dummy_input(backend: str, sz: int) -> np.ndarray:
    if backend in ("rknpu","rknn","rockchip"):
        return np.zeros((1, sz, sz, 3), dtype=np.uint8)     # NHWC uint8
    return np.zeros((1, 3, sz, sz), dtype=np.float32)        # NCHW float32

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",      required=True)
    ap.add_argument("--iters",       type=int, default=100)
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--verbose",     action="store_true")
    args = ap.parse_args()

    print(f"\n{B}{'─'*52}")
    print("  Nuatara Percept — Backend Integration Test")
    print(f"{'─'*52}{X}\n")

    errors = []

    # 1 — Config
    try:
        cfg = EngineConfig.from_yaml(args.config)
        ok("Config loaded", f"{cfg.target}  |  {cfg.backend}  |  {cfg.precision}")
    except Exception as e:
        fail("Config loaded", str(e)); return 1
    if args.verbose: print(f"\n{cfg}\n")

    # 2 — Factory
    try:
        kwargs = {"core": cfg.npu_core_mask()} if cfg.backend in ("rknpu","rknn","rockchip") else {}
        backend = create_backend(cfg.backend, **kwargs)
        ok("Backend created", type(backend).__name__)
    except Exception as e:
        fail("Backend created", str(e)); return 1

    # 3 — Load
    t0 = time.perf_counter()
    try:
        backend.load(cfg.model_path, cfg.precision)
        ok("Model loaded", f"{(time.perf_counter()-t0)*1000:.0f}ms  —  {backend.info}")
    except FileNotFoundError as e:
        fail("Model loaded", str(e))
        print(f"\n  {Y}Hint: check 'model:' path in {args.config}{X}\n"); return 1
    except Exception as e:
        fail("Model loaded", str(e)); return 1

    # 4 — Warmup
    if not args.skip_warmup:
        n = cfg.npu.warmup_iterations
        t0 = time.perf_counter()
        try:
            backend.warmup(n)
            ok("Warmup done", f"{n} iters  {(time.perf_counter()-t0)*1000:.0f}ms total")
        except Exception as e:
            warn("Warmup failed", str(e))
    else:
        warn("Warmup skipped", "first inference may be slow")

    # 5 — Single infer smoke test
    d = dummy_input(cfg.backend, cfg.detection.input_size)
    try:
        outputs = backend.infer(d)
        ok("Inference ok", f"{len(outputs)} tensor(s)  " +
           "  ".join(f"[{i}]{o.shape}" for i,o in enumerate(outputs)))
    except Exception as e:
        fail("Inference ok", str(e)); return 1

    # 6 — Benchmark
    print(f"\n  Running {args.iters} timed iterations...\n")
    latencies = []
    for i in range(args.iters):
        t0 = time.perf_counter()
        try: backend.infer(d)
        except Exception as e:
            fail(f"Iter {i}", str(e)); errors.append(str(e)); break
        latencies.append((time.perf_counter()-t0)*1000)
        if args.verbose: print(f"    {i:4d}  {latencies[-1]:.2f}ms")

    if not latencies:
        print(f"\n{R}FAIL — no successful iterations.{X}\n"); return 1

    sl = sorted(latencies)
    mean = statistics.mean(latencies)
    p95  = sl[max(0, int(.95*len(sl))-1)]
    p99  = sl[max(0, int(.99*len(sl))-1)]
    std  = statistics.stdev(latencies) if len(latencies)>1 else 0.0

    print(f"  {'─'*44}")
    print(f"  {'mean':10s}  {mean:8.2f} ms     {1000/mean:.1f} FPS")
    print(f"  {'median':10s}  {statistics.median(latencies):8.2f} ms")
    print(f"  {'p95':10s}  {p95:8.2f} ms")
    print(f"  {'p99':10s}  {p99:8.2f} ms")
    print(f"  {'stdev':10s}  {std:8.2f} ms")
    print(f"  {'─'*44}")

    if std > mean*0.1: warn("Jitter high", f"stdev {std:.1f}ms > 10% of mean")

    if errors:
        print(f"\n  {R}FAIL{X}  ({len(errors)} error(s))\n")
        for e in errors: print(f"    ✗ {e}")
        return 1

    print(f"\n  {G}{B}PASS ✓{X}\n")
    print(f"  Backend abstraction verified end-to-end.")
    print(f"  To swap hardware: change 'backend:' in {args.config}\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
