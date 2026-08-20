import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path, rows):
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def bootstrap(values, samples=10000, seed=0):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None, None, None
    if len(values) == 1:
        value = float(values[0])
        return value, value, value
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(1)
    return (
        float(values.mean()),
        float(np.percentile(sampled, 2.5)),
        float(np.percentile(sampled, 97.5)),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize GT memory-cleaning replays.")
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    summaries = []
    replacements = []
    frames = []
    for trace in sorted(args.output_root.glob("worldmem_gt_replay_*/access_traces/*.jsonl")):
        run_name = trace.parents[1].name
        for raw in read_jsonl(trace):
            event = raw.get("event")
            row = {"run_name": run_name, **raw}
            if event == "gt_memory_cleaning_replay":
                summaries.append(row)
            elif event == "gt_memory_replay_replacement":
                replacements.append(row)
            elif event == "gt_memory_cleaning_replay_frame":
                frames.append(row)

    if not summaries:
        raise RuntimeError(f"No completed GT memory replays under {args.output_root}")

    invalid = []
    for row in summaries:
        if not row.get("retrieved_frame_identities_match", False):
            invalid.append(f"{row['run_name']}: selected frame identities differ")
        if float(row.get("preintervention_max_abs_diff", float("inf"))) != 0.0:
            invalid.append(f"{row['run_name']}: pre-intervention states differ")
        if float(row.get("non_memory_input_max_abs_diff", float("inf"))) != 0.0:
            invalid.append(f"{row['run_name']}: non-memory inputs differ")
        if int(row.get("replaced_slots", 0)) <= 0:
            invalid.append(f"{row['run_name']}: no generated memory was replaced")
    for row in replacements:
        if row.get("replaced_with_gt") and not row.get("replacement_same_frame_index"):
            invalid.append(f"{row['run_name']}: GT replacement index mismatch")
    if invalid:
        raise RuntimeError("Replay validity checks failed:\n" + "\n".join(invalid))

    aggregate = []
    metrics = ["psnr", "ssim", "lpips"]
    if all("delta_dino_distance" in row for row in summaries):
        metrics.append("dino_distance")
    for metric in metrics:
        values = [float(row[f"delta_{metric}"]) for row in summaries]
        estimate, low, high = bootstrap(values)
        aggregate.append(
            {
                "metric": metric,
                "delta_definition": "GT-cleaned minus generated-memory control",
                "mean_delta": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "replays": len(values),
                "improvement_direction": (
                    "positive" if metric in {"psnr", "ssim"} else "negative"
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "replay_events.csv", summaries)
    write_csv(args.output_dir / "replay_replacements.csv", replacements)
    write_csv(args.output_dir / "replay_frame_metrics.csv", frames)
    write_csv(args.output_dir / "replay_delta_summary.csv", aggregate)
    with (args.output_dir / "replay_validity.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "valid": True,
                "replays": len(summaries),
                "all_frame_identities_match": True,
                "all_preintervention_states_match": True,
                "all_replacements_use_same_frame_index": True,
            },
            handle,
            indent=2,
        )

    print("GT memory-cleaning replay summary")
    for row in aggregate:
        print(
            f"{row['metric']:<14} delta={row['mean_delta']:.6f} "
            f"95% CI=[{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] "
            f"improvement={row['improvement_direction']}"
        )
    print(f"Validity checks: PASS ({len(summaries)} replays)")
    print(f"Wrote: {args.output_dir}")


if __name__ == "__main__":
    main()

