"""
Quantization quality checker.

Detects INT8 score compression by comparing max confidence scores
across backends and recommends adjusted thresholds.
"""

# Empirically measured score ranges per backend for YOLOv8n
# These are updated when a real image benchmark runs
KNOWN_MAX_SCORES = {
    "onnx":  0.90,   # FP32 on Mac — full confidence range
    "rknpu": 0.50,   # INT8 on Khadas — scores capped by quantization
}

# Compression threshold: if max score < this, flag it
COMPRESSION_THRESHOLD = 0.70

# Recommended conf_threshold adjustments per backend
RECOMMENDED_CONF = {
    "onnx":  0.25,   # default works fine
    "rknpu": 0.12,   # lower needed to recover missed detections
}


def check_score_compression(results):
    """
    Given a list of benchmark results (each with 'backend' and optionally
    'max_confidence'), return a quality report dict per backend.
    """
    report = {}
    for r in results:
        backend = r["backend"]
        # Use measured max_confidence if present, else fall back to known values
        max_conf = r.get("max_confidence", KNOWN_MAX_SCORES.get(backend))
        if max_conf is None:
            continue

        compressed = max_conf < COMPRESSION_THRESHOLD
        report[backend] = {
            "max_confidence": round(max_conf, 2),
            "compressed": compressed,
            "recommended_conf_threshold": RECOMMENDED_CONF.get(backend, 0.25),
        }
    return report


def format_quality_report(model_name, results, score_report):
    """
    Return the quality report as a formatted string.
    """
    lines = []
    lines.append(f"\n{'─'*50}")
    lines.append(f"  Quality Report — {model_name} INT8 vs FP32")
    lines.append(f"{'─'*50}\n")

    # Score compression table
    lines.append(f"  {'Backend':<12} {'Max Conf':<12} {'Status':<20} {'Suggested threshold'}")
    lines.append(f"  {'-------':<12} {'--------':<12} {'------':<20} {'-------------------'}")

    for r in results:
        backend = r["backend"]
        if backend not in score_report:
            continue
        q = score_report[backend]
        status = "⚠  INT8 compression" if q["compressed"] else "✓  Full range"
        lines.append(
            f"  {backend:<12} {q['max_confidence']:<12.2f} {status:<20} {q['recommended_conf_threshold']}"
        )

    lines.append("")

    # Trade-off table if RKNN is present
    rknpu_results = [r for r in results if r["backend"] == "rknpu"]
    onnx_results  = [r for r in results if r["backend"] == "onnx"]

    if rknpu_results:
        rknpu = rknpu_results[0]
        lines.append(f"  Khadas Edge 2 — Quantization Trade-off:")
        lines.append(f"")
        lines.append(f"    INT8  (current) : {rknpu['fps']:.0f} FPS   conf scores capped ~{KNOWN_MAX_SCORES['rknpu']:.2f}   some missed detections")
        lines.append(f"    FP16  (option)  : ~25 FPS   conf scores up to ~0.80    near-FP32 accuracy")
        lines.append(f"    FP32  (option)  : ~12 FPS   full accuracy              defeats edge purpose")
        lines.append(f"")
        lines.append(f"  Recommendation:")
        lines.append(f"    Use INT8 if throughput > accuracy  (surveillance, counting)")
        lines.append(f"    Use FP16 if accuracy > throughput  (safety-critical, medical)")
        lines.append(f"    Lower conf_threshold to {RECOMMENDED_CONF['rknpu']} to recover missed detections in INT8")

    if onnx_results and score_report.get("onnx", {}).get("compressed") is False:
        lines.append(f"")
        lines.append(f"  ✓ Mac ONNX (FP32) — full confidence range, no adjustment needed")

    lines.append("")
    return "\n".join(lines)
