"""
nuatara CLI — entry point for all commands.

Usage:
    nuatara recommend <model>
    nuatara benchmark <model> --config <config.yaml>
"""

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


def main():
    parser = argparse.ArgumentParser(
        prog="nuatara",
        description="Hardware-aware AI deployment advisor"
    )
    sub = parser.add_subparsers(dest="command")

    # nuatara recommend <model>
    rec = sub.add_parser("recommend", help="Recommend best runtime for a model")
    rec.add_argument("model", help="Model name or path (e.g. yolov8n)")

    # nuatara benchmark <model> --config <yaml>
    bench = sub.add_parser("benchmark", help="Benchmark a model on a specific config")
    bench.add_argument("model", help="Model name or path")
    bench.add_argument("--config", required=True, help="Path to device config YAML")

    args = parser.parse_args()

    if args.command == "recommend":
        cmd_recommend(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
