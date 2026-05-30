"""
scripts/benchmark.py

End-to-end benchmark of the Tamakkan pipeline on a video clip.
Produces a markdown table suitable for the graduation report.

Measures:
- Wall-clock FPS (full pipeline)
- Per-model latency (mean, p50, p95, p99)
- Memory footprint (RAM + GPU)
- Event counts (proves detectors fire)
- Determinism (run twice, compare)

Usage:
    PYTHONPATH=src:third_party python scripts/benchmark.py path/to/clip.mp4
"""
import sys
import time
import statistics
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch

_repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo / "src"))
sys.path.insert(0, str(_repo / "third_party"))

from tamakkan.pipeline import TamakkanPipeline


def get_jetson_stats():
    """Read RAM and GPU memory at this instant."""
    stats = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    stats["ram_total_mb"] = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    stats["ram_avail_mb"] = int(line.split()[1]) // 1024
        stats["ram_used_mb"] = stats["ram_total_mb"] - stats["ram_avail_mb"]
    except Exception:
        pass
    try:
        # tegrastats one-shot
        import subprocess
        out = subprocess.run(
            ["tegrastats", "--interval", "100", "--logfile", "/dev/stdout"],
            capture_output=True, text=True, timeout=0.5
        )
        if out.stdout:
            stats["tegrastats"] = out.stdout.split("\n")[0]
    except Exception:
        pass
    return stats


def percentile(values, p):
    """p in [0, 100]."""
    if not values:
        return 0.0
    s = sorted(values)
    k = int(len(s) * p / 100)
    return s[min(k, len(s) - 1)]


def run_benchmark(video_path: str, label: str):
    """Run the pipeline on a clip, return per-stage timings and event counts."""

    weights_dir = _repo / "weights"

    # Auto-detect engines (Jetson) vs pth (PC).
    def pick(engine_name, pth_name):
        e = weights_dir / engine_name
        return str(e if e.exists() else weights_dir / pth_name)

    print(f"\n{'='*70}")
    print(f"BENCHMARK: {label}")
    print(f"Video: {video_path}")
    print(f"{'='*70}")

    # Time the pipeline load (cold start cost)
    t_load_start = time.perf_counter()
    pipeline = TamakkanPipeline(
        yolo_weights=pick("best.engine", "best.pt"),
        bytetrack_config=str(weights_dir / "bytetrack_tamakkan.yaml"),
        depth_weights=pick("depth_anything_v2_vits.engine", "depth_anything_v2_vits.pth"),
        lane_weights=pick("culane_res18_v2.engine", "culane_res18_v2.pth"),
        device="cuda:0",
        depth_every_n=8,
        lanes_every_n=5,
        ocr_frame_skip=999,
    )
    t_load = time.perf_counter() - t_load_start
    print(f"Pipeline load time: {t_load:.2f}s")

    stats_before = get_jetson_stats()

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source: {total_frames} frames, {src_fps:.1f} fps, {width}x{height}")

    # Per-stage timings
    stage_times = defaultdict(list)
    frame_times = []
    events = []

    # Warm up — first frame includes lazy engine init
    print("Warming up (3 frames)...")
    for _ in range(3):
        ok, frame = cap.read()
        if not ok:
            break
        _ = pipeline.process_frame(frame)

    cap.release()
    cap = cv2.VideoCapture(video_path)

    # Actual timed run
    print("Starting timed run...")
    t_run_start = time.perf_counter()
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t_frame_start = time.perf_counter()
        result = pipeline.process_frame(frame)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_frame_end = time.perf_counter()

        frame_times.append((t_frame_end - t_frame_start) * 1000)  # ms

        # Try to extract per-stage timings if the pipeline records them
        if hasattr(result, "stage_times"):
            for stage, ms in result.stage_times.items():
                stage_times[stage].append(ms)

        # Collect events
        if hasattr(result, "events") and result.events:
            for ev in result.events:
                events.append({
                    "frame": frame_idx,
                    "type": getattr(ev, "event_type", "unknown"),
                    "subtype": getattr(ev, "subtype", None),
                })

        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.perf_counter() - t_run_start
            print(f"  {frame_idx} frames, {frame_idx/elapsed:.1f} wall-FPS")

    t_run = time.perf_counter() - t_run_start
    cap.release()

    stats_after = get_jetson_stats()

    # Compute metrics
    wall_fps = frame_idx / t_run
    mean_ms = statistics.mean(frame_times)
    median_ms = statistics.median(frame_times)
    p95_ms = percentile(frame_times, 95)
    p99_ms = percentile(frame_times, 99)
    max_ms = max(frame_times)
    min_ms = min(frame_times)

    print(f"\n--- Results ---")
    print(f"Frames processed:    {frame_idx}")
    print(f"Wall seconds:        {t_run:.1f}")
    print(f"Wall-FPS:            {wall_fps:.2f}")
    print(f"Per-frame latency:")
    print(f"  mean:              {mean_ms:.1f} ms")
    print(f"  median:            {median_ms:.1f} ms")
    print(f"  p95:               {p95_ms:.1f} ms")
    print(f"  p99:               {p99_ms:.1f} ms")
    print(f"  min/max:           {min_ms:.1f} / {max_ms:.1f} ms")
    if stage_times:
        print(f"Per-stage latency (mean ms):")
        for stage, ts in sorted(stage_times.items()):
            print(f"  {stage:20s}  mean={statistics.mean(ts):.1f}  p95={percentile(ts, 95):.1f}")

    print(f"Events fired: {len(events)}")
    event_counts = defaultdict(int)
    for ev in events:
        event_counts[ev["type"]] += 1
    for typ, n in sorted(event_counts.items()):
        print(f"  {typ:20s}  {n}")

    if "ram_used_mb" in stats_before and "ram_used_mb" in stats_after:
        print(f"RAM at start:  {stats_before['ram_used_mb']} MB used")
        print(f"RAM at end:    {stats_after['ram_used_mb']} MB used")

    return {
        "label": label,
        "load_time_s": t_load,
        "frames": frame_idx,
        "wall_seconds": t_run,
        "wall_fps": wall_fps,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "stage_times": {k: list(v) for k, v in stage_times.items()},
        "events": events,
        "event_counts": dict(event_counts),
        "ram_used_mb_start": stats_before.get("ram_used_mb"),
        "ram_used_mb_end": stats_after.get("ram_used_mb"),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/benchmark.py <video_path>")
        sys.exit(1)

    video = sys.argv[1]

    # Run twice for determinism check
    result1 = run_benchmark(video, "Run 1")
    result2 = run_benchmark(video, "Run 2 (determinism check)")

    if result1 and result2:
        print(f"\n{'='*70}")
        print("DETERMINISM CHECK")
        print(f"{'='*70}")
        e1 = sorted([(e["frame"], e["type"]) for e in result1["events"]])
        e2 = sorted([(e["frame"], e["type"]) for e in result2["events"]])
        if e1 == e2:
            print(f"✓ Both runs fired identical events ({len(e1)} events)")
        else:
            print(f"✗ Events differ between runs:")
            print(f"  Run 1: {len(e1)} events")
            print(f"  Run 2: {len(e2)} events")
            diff = set(e1) ^ set(e2)
            for ev in sorted(diff)[:20]:
                print(f"    {ev}")
