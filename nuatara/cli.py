import argparse
import sys
from nuatara.discover import discover_devices
from nuatara.benchmark import run_benchmark
from nuatara.score import compute_scores
from nuatara.report import print_report


def cmd_recommend(args):
    model = args.model
    print(f"\nNuatara — feasibility check for: {model}\n")

    devices = discover_devices()
    if not devices:
        print("No supported devices found.")
        sys.exit(1)

    results = []
    for device in devices:
        print(f"  Benchmarking on {device['name']}...")
        result = run_benchmark(model, device)
        if result:
            results.append(result)

    if not results:
        print("No successful benchmark results.")
        sys.exit(1)

    scored = compute_scores(results)
    print_report(model, scored)


def cmd_benchmark(args):
    from nuatara.benchmark import run_benchmark_from_config
    run_benchmark_from_config(args.config)


def cmd_demo(args):
    from nuatara.demo import run_demo
    run_demo(args.model, args.video, backend_name=args.backend, output_path=args.output)


def main():
    parser = argparse.ArgumentParser(
        prog="nuatara",
        description="Hardware-aware AI deployment advisor"
    )
    sub = parser.add_subparsers(dest="command")

    rec = sub.add_parser("recommend", help="Recommend best runtime for a model")
    rec.add_argument("model", help="Model name (e.g. yolov8n)")

    bench = sub.add_parser("benchmark", help="Benchmark a model on a specific config")
    bench.add_argument("model", help="Model name")
    bench.add_argument("--config", required=True, help="Path to device config YAML")

    demo = sub.add_parser("demo", help="Run detection on a video with live preview")
    demo.add_argument("model", help="Model name (e.g. yolov8n)")
    demo.add_argument("--video", required=True, help="Path to video file")
    demo.add_argument("--output", default=None, help="Output video path")
    demo.add_argument("--backend", default="onnx", help="Backend: onnx or rknpu")

    args = parser.parse_args()

    if args.command == "recommend":
        cmd_recommend(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
