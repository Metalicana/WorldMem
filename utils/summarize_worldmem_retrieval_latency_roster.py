"""Build full-rollout ms/query table rows for the WorldMem B32 roster."""

import argparse
import json
import math
from pathlib import Path

from utils.summarize_worldmem_retrieval_latency import read_profile, summarize, write_csv


LABELS = {"unbounded": "WorldMem", "fifo": "WorldMem + FIFO",
          "mce": "WorldMem + MCE", "kcenter_coreset": "WorldMem + K-center",
          "rarity_irreplaceability": "WorldMem + RI",
          "slam_covisibility": "WorldMem + KEEPSAKE"}
DEFAULT_SPECS = "unbounded: fifo:32 mce:32 kcenter_coreset:32 rarity_irreplaceability:32 slam_covisibility:32"


def collect(root, specs, videos, seed):
    output, identities = [], []
    settings = set()
    for i, spec in enumerate(specs.split()):
        policy, budget = spec.split(":")
        budget = int(budget) if budget else None
        if policy not in LABELS or (policy == "unbounded") != (budget is None):
            raise ValueError(f"Invalid policy/budget specification: {spec}")
        tag = "" if budget is None else f"_b{budget}"
        name = f"worldmem_retrieval_latency_{policy}{tag}_60s_n{videos}_seed{seed}"
        trace = root / name / "access_traces" / f"{name}.jsonl"
        trajectories = read_profile(trace, policy, budget, videos, 600)
        if set(trajectories) != set(range(videos)):
            raise ValueError(f"Unexpected trajectory IDs for {name}")
        for batch, rows in trajectories.items():
            for row in rows:
                device = row.get("cuda_device_name")
                samples = row.get("retrieval_fov_samples")
                if not device or samples != 10000:
                    raise ValueError("Expected logged GPU name and native 10000-sample retrieval")
                if (row.get("memory_bank_device") != "cpu"
                        or row.get("memory_reference_source") != "predicted"
                        or row.get("memory_feature_backend") != "latent"
                        or row.get("retrieval_candidate_cap") is not None
                        or row.get("generation_seed") != seed + batch):
                    raise ValueError("Profile does not match CPU/predicted/latent/uncapped/seed protocol")
                expected_candidates = 600 + int(row["rollout_frame"]) if budget is None else budget
                if int(row["candidate_count"]) != expected_candidates:
                    raise ValueError("Unexpected candidate count for the matched native retrieval")
                value = float(row["query_milliseconds"])
                if not math.isfinite(value) or value < 0:
                    raise ValueError("Invalid query time")
                settings.add((device, samples))
        result = summarize(LABELS[policy], policy, budget, trajectories, 10,
                           [(0, 60), (0, 15), (45, 60)], seed + i * 100)
        for row in result:
            row.update(run_name=name, trace_path=str(trace))
        output.extend(result)
        identities.append({"run_name": name, "trace_path": str(trace),
                           "trajectory_ids": sorted(trajectories)})
    if len(settings) != 1:
        raise ValueError("Profiles used different GPU models or retrieval settings")
    return output, identities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-videos", type=int, default=15)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--policy-specs", default=DEFAULT_SPECS)
    args = parser.parse_args()
    rows, identities = collect(args.output_root, args.policy_specs, args.expected_videos, args.seed)
    directory = args.output_root / "summary"
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(directory / "worldmem_retrieval_latency_windows.csv", rows)
    full = [row for row in rows if row["generated_window"] == "0-60s"]
    path = directory / "worldmem_retrieval_latency_main.csv"
    write_csv(path, full)
    (directory / "protocol.json").write_text(json.dumps({
        "scope": "generate_condition_indices; synchronized candidate scoring and selection",
        "excluded": ["candidate-list construction", "latent gathering/transfer", "denoising",
                     "encoding/decoding", "bank updates", "trace-file writing"],
        "aggregation": "mean per trajectory over 600 queries, then equal-weight trajectory mean",
        "ci": "95% trajectory bootstrap; one-trajectory pilots have degenerate intervals",
        "bank_device": "cpu (verified profile metadata)", "feature_backend": "latent (verified profile metadata)",
        "nominal_rollout_seconds": 60, "generated_frames": 600,
        "traces": identities,
        "limitations": "Matching GPU model is not proof of an idle GPU or identical clocks; run without competing GPU jobs. These are separate latency runs, not reconstructed timings of historical quality runs."
    }, indent=2) + "\n")
    for row in full:
        print(f"{row['policy_label']:<24} {row['query_time_ms_trajectory_mean']:8.3f} ms/query "
              f"[{row['query_time_ms_ci95_low']:.3f}, {row['query_time_ms_ci95_high']:.3f}] "
              f"n={row['trajectories']}")
    print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
