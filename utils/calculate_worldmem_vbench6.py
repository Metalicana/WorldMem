"""Calculate the normalized six-dimension VBench custom-input aggregate."""

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.build_worldmem_final_metric_status import load_vbench_run  # noqa: E402


SPEC = {
    "subject_consistency": (0.1462, 1.0, 1.0),
    "background_consistency": (0.2615, 1.0, 1.0),
    "motion_smoothness": (0.7060, 0.9975, 1.0),
    "dynamic_degree": (0.0, 1.0, 0.5),
    "aesthetic_quality": (0.0, 1.0, 1.0),
    "imaging_quality": (0.0, 1.0, 1.0),
}

RUNS = (
    ("worldmem_unbounded_60s_n30", "WorldMem", ""),
    ("worldmem_fifo_b32_60s_n30", "WorldMem + FIFO", 32),
    ("worldmem_mce_b32_60s_n15", "WorldMem + MCE", 32),
    ("worldmem_kcenter_coreset_b32_60s_n15", "WorldMem + K-center", 32),
    ("worldmem_rarity_irreplaceability_b32_60s_n30", "WorldMem + RI", 32),
    ("worldmem_slam_covisibility_b32_60s_n30", "WorldMem + KEEPSAKE", 32),
)


def calculate_vbench6(values):
    if set(values) != set(SPEC):
        missing = sorted(set(SPEC) - set(values))
        extra = sorted(set(values) - set(SPEC))
        raise ValueError(f"VBench-6 dimension mismatch: missing={missing}, extra={extra}")
    weighted = 0.0
    total_weight = 0.0
    for dimension, (low, high, weight) in SPEC.items():
        value = float(values[dimension])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid {dimension} score: {value}")
        weighted += weight * (value - low) / (high - low)
        total_weight += weight
    return 100.0 * weighted / total_weight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for run_name, label, budget in RUNS:
        values, source = load_vbench_run(args.root, run_name, args.limit)
        if not values:
            raise RuntimeError(
                f"{run_name}: missing, stale, or unmatched {args.limit}-video VBench result"
            )
        score = calculate_vbench6(values)
        rows.append({
            "run_name": run_name,
            "model": label,
            "budget": budget,
            "videos": args.limit,
            **values,
            "vbench6_percent": score,
            "source": str(source),
        })

    print(f"{'MODEL':<28} {'VBENCH-6 (%)':>14}")
    for row in rows:
        print(f"{row['model']:<28} {row['vbench6_percent']:>14.2f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
