"""
Scoring formula: rank results across devices.
Score = 0.4 * throughput + 0.3 * memory_efficiency + 0.2 * setup + 0.1 * power
For now we score on throughput only (others default to 0.5).
"""


def compute_scores(results):
    if not results:
        return []

    max_fps = max(r["fps"] for r in results)

    scored = []
    for r in results:
        throughput_norm = r["fps"] / max_fps if max_fps > 0 else 0
        score = round(
            0.4 * throughput_norm * 100
            + 0.3 * 50   # memory_efficiency placeholder
            + 0.2 * 50   # setup_complexity placeholder
            + 0.1 * 50,  # power placeholder
            0
        )
        scored.append({**r, "score": int(score)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored
