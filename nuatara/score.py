"""
Scoring formula:
Score = 0.4 * throughput + 0.3 * memory_efficiency + 0.2 * setup + 0.1 * power
"""

# Memory budget per device (MB) — higher budget = more headroom
MEMORY_BUDGET = {
    "onnx": 500,    # MacBook — plenty of RAM
    "rknpu": 200,   # Khadas — tighter
}

# Setup complexity score (0-100) — how hard is it to get running
SETUP_SCORE = {
    "onnx": 90,     # pip install onnxruntime, done
    "rknpu": 55,    # need RKNN SDK, model conversion, board setup
}

# Power estimate score (0-100) — higher = more power efficient
POWER_SCORE = {
    "onnx": 60,     # MacBook ANE is efficient but still a laptop
    "rknpu": 85,    # Khadas NPU — low power edge device
}


def compute_scores(results):
    if not results:
        return []

    max_fps = max(r["fps"] for r in results)

    scored = []
    for r in results:
        backend = r["backend"]

        throughput_norm = (r["fps"] / max_fps) * 100 if max_fps > 0 else 0

        budget = MEMORY_BUDGET.get(backend, 500)
        mem_used = r.get("memory_mb", 15.6)
        memory_score = max(0, min(100, (1 - mem_used / budget) * 100))

        setup = SETUP_SCORE.get(backend, 50)
        power = POWER_SCORE.get(backend, 50)

        score = round(
            0.4 * throughput_norm
            + 0.3 * memory_score
            + 0.2 * setup
            + 0.1 * power,
            0
        )
        scored.append({**r, "score": int(score)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored
