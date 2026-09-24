#!/usr/bin/env python3
"""Summarize restart-safe WorldMem native evaluation traces and rFID files."""

import argparse
import csv
import json
from pathlib import Path


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def read_batch_metrics(trace_path, limit):
    by_batch = {}
    with trace_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at {trace_path}:{line_number}"
                ) from exc
            if record.get("event") != "batch_metrics":
                continue
            batch_idx = int(record["global_batch_idx"])
            if 0 <= batch_idx < limit:
                by_batch[batch_idx] = record

    expected = set(range(limit))
    missing = sorted(expected - set(by_batch))
    if missing:
        raise RuntimeError(
            f"{trace_path} is missing batch_metrics for batches: {missing}"
        )
    return [by_batch[index] for index in range(limit)]


def read_rfid(path):
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("FID Score:"):
            return float(line.split(":", 1)[1].strip())
    raise RuntimeError(f"Could not parse FID score from {path}")


def mean(records, key):
    return sum(float(record[key]) for record in records) / len(records)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--rfid-videos", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    runs = parse_csv(args.runs)
    labels = parse_csv(args.labels)
    if len(runs) != len(labels):
        raise ValueError("--runs and --labels must contain the same number of entries")
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.rfid_videos is not None and not 1 <= args.rfid_videos <= args.limit:
        raise ValueError("--rfid-videos must be between 1 and --limit")

    rows = []
    for run_name, label in zip(runs, labels):
        run_dir = args.output_root / run_name
        trace_path = run_dir / "access_traces" / f"{run_name}.jsonl"
        if not trace_path.exists():
            raise FileNotFoundError(f"Missing access trace: {trace_path}")
        records = read_batch_metrics(trace_path, args.limit)
        pred_dir = run_dir / "videos" / "test_vis" / "pred"
        gt_dir = run_dir / "videos" / "test_vis" / "gt"
        pred_videos = sorted(pred_dir.glob("video_batch*.mp4"))
        gt_videos = sorted(gt_dir.glob("video_batch*.mp4"))
        if len(pred_videos) < args.limit or len(gt_videos) < args.limit:
            raise RuntimeError(
                f"{run_name}: expected at least {args.limit} prediction/GT videos; "
                f"found {len(pred_videos)}/{len(gt_videos)}"
            )

        rows.append(
            {
                "run_name": run_name,
                "method": label,
                "videos": args.limit,
                "generated_frames_per_video": 100,
                "rfid_videos": args.rfid_videos,
                "rfid_frames": (
                    args.rfid_videos * 100
                    if args.rfid_videos is not None
                    else None
                ),
                "psnr": mean(records, "psnr"),
                "lpips": mean(records, "lpips"),
                "mse": mean(records, "mse"),
                "rfid": read_rfid(run_dir / "videos" / "test_vis" / "fid_results.txt"),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "summary.csv"
    json_path = args.output_dir / "summary.json"
    write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(
            {
                "protocol": {
                    "name": "WorldMem native beyond-context-window evaluation",
                    "initial_memory_frames": 600,
                    "generated_frames": 100,
                    "generator_context_frames": 8,
                    "retrieved_memory_frames": 8,
                    "sampling_timesteps": 20,
                    "gt_target": "VAE-reconstructed exact-index GT",
                    "metric_videos": args.limit,
                    "rfid_videos": args.rfid_videos,
                    "rfid_frames": (
                        args.rfid_videos * 100
                        if args.rfid_videos is not None
                        else None
                    ),
                },
                "results": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nWorldMem native beyond-context results")
    print(f"{'Method':<24} {'N':>4} {'PSNR':>10} {'LPIPS':>10} {'rFID':>10}")
    for row in rows:
        rfid = "pending" if row["rfid"] is None else f"{row['rfid']:.4f}"
        print(
            f"{row['method']:<24} {row['videos']:>4d} "
            f"{row['psnr']:>10.4f} {row['lpips']:>10.6f} {rfid:>10}"
        )
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
