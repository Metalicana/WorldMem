"""Build the locked six-policy WorldMem metric-completion table."""

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path


LOCKED_RUNS = (
    "worldmem_unbounded_60s_n30",
    "worldmem_fifo_b32_60s_n30",
    "worldmem_rarity_irreplaceability_b32_60s_n30",
    "worldmem_slam_covisibility_b32_60s_n30",
    "worldmem_kcenter_coreset_b32_60s_n15",
    "worldmem_mce_b32_60s_n15",
)
DURATIONS = (10, 20, 30, 60)
DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
)
VIDEO_RE = re.compile(r"^video_batch(?P<batch>\d+)_0_rank0(?:_step.*)?\.mp4$")


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_result(run_dir):
    candidates = sorted(Path(run_dir).glob("*_eval_results.json"))
    return candidates[-1] if candidates else None


def discover_matched_batches(run_dir, limit):
    pred_dir = Path(run_dir) / "videos" / "test_vis" / "pred"
    batches = set()
    if pred_dir.is_dir():
        for path in pred_dir.glob("video_batch*.mp4"):
            match = VIDEO_RE.match(path.name)
            if match and path.stat().st_size > 4096:
                batches.add(int(match.group("batch")))
    required = list(range(int(limit)))
    return required if all(batch in batches for batch in required) else []


def load_prefix_metric(summary_path, metric, runs, limit):
    values = {}
    source = Path(summary_path)
    if not source.is_file():
        return values
    with source.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            run = row.get("run_name")
            if run not in runs:
                continue
            duration = int(row["duration_sec"])
            complete = int(float(row.get("completed_videos") or 0))
            failed = int(float(row.get("failed_videos") or 0))
            videos = int(float(row.get("videos") or 0))
            value = row.get(metric)
            if complete >= limit and failed == 0 and videos == limit and value not in {None, ""}:
                values[(run, duration)] = float(value)
    return values


def load_vbench_run(root, run, limit):
    run_dir = Path(root) / run
    result_path = latest_result(run_dir)
    manifest_path = run_dir / "input_selection.json"
    if result_path is None or not manifest_path.is_file():
        return {}, None
    manifest = load_json(manifest_path)
    if manifest.get("selected_batch_ids") != list(range(limit)):
        return {}, None
    payload = load_json(result_path)
    scores = {}
    for dimension in DIMENSIONS:
        value = payload.get(dimension)
        if not isinstance(value, list) or len(value) != 2:
            return {}, None
        details = value[1]
        if not isinstance(details, list) or len(details) != limit:
            return {}, None
        scores[dimension] = float(value[0])
    return scores, result_path


def load_cut3r(summary_path, valid, runs, limit):
    values = {}
    source = Path(summary_path)
    if not valid or not source.is_file():
        return values
    with source.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            run = row.get("run_name")
            if run not in runs or int(float(row.get("videos") or 0)) < limit:
                continue
            values[run] = {
                "rotation": row.get("rotation_error_deg_mean_mean"),
                "translation": row.get("translation_error_scale_only_mean_mean"),
                "score": row.get("worldscore_camera_control_score_mean"),
            }
    return values


def git_commit(repo_root):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--lpips-summary", type=Path, required=True)
    parser.add_argument("--fvd-summary", type=Path, required=True)
    parser.add_argument("--vbench-root", type=Path, required=True)
    parser.add_argument("--vbench-long-root", type=Path, required=True)
    parser.add_argument("--cut3r-summary", type=Path, required=True)
    parser.add_argument("--cut3r-validity", type=Path, required=True)
    parser.add_argument("--runs", default=",".join(LOCKED_RUNS))
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()

    runs = tuple(run.strip() for run in args.runs.split(",") if run.strip())
    required_ids = list(range(args.limit))
    lpips = load_prefix_metric(args.lpips_summary, "lpips", runs, args.limit)
    fvd = load_prefix_metric(args.fvd_summary, "fvd", runs, args.limit)

    cut3r_valid = False
    if args.cut3r_validity.is_file():
        validity = load_json(args.cut3r_validity)
        cut3r_valid = validity.get("valid") is True
    cut3r = load_cut3r(args.cut3r_summary, cut3r_valid, runs, args.limit)

    rows = []
    sources = {}
    for run in runs:
        batches = discover_matched_batches(args.output_root / run, args.limit)
        vbench, vbench_source = load_vbench_run(args.vbench_root, run, args.limit)
        vbench_long, vbench_long_source = load_vbench_run(
            args.vbench_long_root,
            run,
            args.limit,
        )
        row = {
            "run_name": run,
            "videos_matched": len(batches),
        }
        for duration in DURATIONS:
            row[f"lpips_{duration}"] = lpips.get((run, duration), "missing")
            row[f"fvd_{duration}"] = fvd.get((run, duration), "missing")
        for short, dimension in zip(
            ("subject", "background", "motion", "dynamic", "aesthetic", "imaging"),
            DIMENSIONS,
        ):
            row[f"vbench_{short}"] = vbench.get(dimension, "missing")
            row[f"vbench_long_{short}"] = vbench_long.get(dimension, "missing")
        cut3r_row = cut3r.get(run, {})
        row["cut3r_valid"] = bool(cut3r_valid and cut3r_row)
        row["cut3r_rotation_error"] = cut3r_row.get("rotation", "invalid")
        row["cut3r_translation_error"] = cut3r_row.get("translation", "invalid")
        row["cut3r_camera_control_score"] = cut3r_row.get("score", "invalid")
        rows.append(row)
        sources[run] = {
            "selected_batch_ids": batches,
            "required_batch_ids": required_ids,
            "vbench_result": None if vbench_source is None else str(vbench_source),
            "vbench_long_result": (
                None if vbench_long_source is None else str(vbench_long_source)
            ),
        }

    fields = list(rows[0]) if rows else []
    csv_path = args.metrics_dir / "worldmem_final_metric_status.csv"
    json_path = args.metrics_dir / "worldmem_final_metric_status.json"
    write_csv(csv_path, rows, fields)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol": "worldmem_locked_b32_first15_v1",
                "limit": args.limit,
                "required_batch_ids": required_ids,
                "git_commit": git_commit(args.repo_root),
                "cut3r_validity_path": str(args.cut3r_validity),
                "cut3r_valid": cut3r_valid,
                "metric_sources": {
                    "lpips": str(args.lpips_summary),
                    "fvd": str(args.fvd_summary),
                    "vbench": str(args.vbench_root),
                    "vbench_long": str(args.vbench_long_root),
                    "cut3r": str(args.cut3r_summary),
                },
                "run_sources": sources,
                "rows": rows,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    for row in rows:
        missing = sum(value in {"missing", "invalid"} for value in row.values())
        print(f"{row['run_name']}: videos={row['videos_matched']} missing_or_invalid={missing}")


if __name__ == "__main__":
    main()
