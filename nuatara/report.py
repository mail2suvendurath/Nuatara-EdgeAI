"""
Format and print the recommendation report.
"""


def print_report(model_name, scored):
    print(f"\n{'='*50}")
    print(f"  nuatara recommend {model_name}")
    print(f"{'='*50}\n")

    for r in scored:
        print(f"  {r['device']}")
        print(f"    Runtime  : {r['backend'].upper()}")
        if r.get("tokens_per_sec"):
            print(f"    Throughput: {r['tokens_per_sec']} tokens/sec")
        else:
            print(f"    Latency  : {r['latency_ms']}ms   {r['fps']} FPS")
        print(f"    Score    : {r['score']}")
        print()

    if scored:
        best = scored[0]
        print(f"  Recommendation: {best['device']}")
        print(f"  → Best score ({best['score']}) with {best['latency_ms']}ms latency\n")
