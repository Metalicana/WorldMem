"""Build CPU-only galleries of extreme WorldMem retrieval failures.

Two complementary views are emitted:

``actual_history``
    Each policy's selected frame is read from that policy's own generated
    rollout. This visualizes the memory content the model actually consumed.

``common_source``
    Both policies' selected frame identities are read from the same Unbounded
    rollout. This isolates retrieval choice from policy-specific rollout drift.

The trace metrics used by ``actual_history`` were measured from the stored
latent after VAE decoding during generation. The common-source PSNR/SSIM values
are recomputed from MP4 pixels on CPU against exact-index dataset ground truth.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.worldmem_eval_common import (  # noqa: E402
    list_prediction_videos,
    resolve_dataset_video_for_batch,
)


TILE_WIDTH = 292
IMAGE_HEIGHT = 180
HEADER_HEIGHT = 43
FOOTER_HEIGHT = 64
TILE_HEIGHT = HEADER_HEIGHT + IMAGE_HEIGHT + FOOTER_HEIGHT

COLORS = {
    "target": (45, 104, 176),
    "unbounded": (194, 61, 56),
    "bounded": (39, 145, 83),
    "ground_truth": (90, 96, 106),
}


def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def mean(values):
    values = [finite_float(value) for value in values]
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_trace(run_dir):
    paths = sorted((Path(run_dir) / "access_traces").glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No access trace under {run_dir}")
    return paths[0]


def load_quality_trace(path):
    """Return deduplicated retrieval rows and per-batch run metadata."""
    metadata = {}
    retrieved = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] malformed trace row {path}:{line_number}: {exc}")
                continue
            event = row.get("event")
            batch = row.get("global_batch_idx", row.get("batch_idx"))
            if batch is None:
                continue
            batch = int(batch)
            if event == "memory_run_start":
                metadata[batch] = row
            elif event == "retrieved_memory_quality":
                key = (
                    batch,
                    int(row["target_frame"]),
                    int(row.get("context_slot", -1)),
                )
                # Resumed jobs can append a repeated batch. The last complete
                # exposure is the one represented by the final saved video.
                retrieved[key] = row

    grouped = defaultdict(list)
    for (batch, target_frame, _slot), row in retrieved.items():
        grouped[(batch, target_frame)].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row.get("context_slot", -1)))
    return dict(grouped), metadata


def validate_matched_runs(reference_metadata, policy_metadata, matched_batches):
    for batch in matched_batches:
        reference = reference_metadata.get(batch)
        policy = policy_metadata.get(batch)
        if reference is None or policy is None:
            raise RuntimeError(f"Missing memory_run_start metadata for batch {batch}")
        for key in ("dataset_batch_idx", "generation_seed", "context_frames"):
            if reference.get(key) != policy.get(key):
                raise RuntimeError(
                    f"Matched-run identity mismatch for batch {batch}, {key}: "
                    f"{reference.get(key)!r} != {policy.get(key)!r}"
                )


def generated_rows(rows, context_frames):
    return [
        row
        for row in rows
        if not bool(row.get("source_is_initial_context"))
        and int(row.get("selected_memory_frame", -1)) >= int(context_frames)
    ]


def summarize_trace_items(rows):
    return {
        "items": len(rows),
        "psnr": mean(row.get("decoded_memory_psnr") for row in rows),
        "ssim": mean(row.get("decoded_memory_ssim") for row in rows),
        "lpips": mean(row.get("decoded_memory_lpips") for row in rows),
    }


def worst_trace_item(rows):
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            finite_float(row.get("decoded_memory_lpips")) or -math.inf,
            -(finite_float(row.get("decoded_memory_psnr")) or math.inf),
            -int(row.get("context_slot", -1)),
        ),
    )


def build_actual_candidates(
    reference_groups,
    policy_groups,
    context_frames,
    fps,
    late_start_sec,
):
    rows = []
    for batch, target_frame in sorted(set(reference_groups) & set(policy_groups)):
        horizon_sec = (target_frame - int(context_frames)) / float(fps)
        if horizon_sec < float(late_start_sec):
            continue
        reference_items = generated_rows(
            reference_groups[(batch, target_frame)], context_frames
        )
        policy_items = generated_rows(
            policy_groups[(batch, target_frame)], context_frames
        )
        if not reference_items or not policy_items:
            continue
        reference_summary = summarize_trace_items(reference_items)
        policy_summary = summarize_trace_items(policy_items)
        reference_worst = worst_trace_item(reference_items)
        policy_worst = worst_trace_item(policy_items)
        rows.append(
            {
                "batch_idx": batch,
                "target_frame": target_frame,
                "horizon_sec": horizon_sec,
                "unbounded_generated_items": reference_summary["items"],
                "bounded_generated_items": policy_summary["items"],
                "unbounded_mean_psnr": reference_summary["psnr"],
                "unbounded_mean_ssim": reference_summary["ssim"],
                "unbounded_mean_lpips": reference_summary["lpips"],
                "bounded_mean_psnr": policy_summary["psnr"],
                "bounded_mean_ssim": policy_summary["ssim"],
                "bounded_mean_lpips": policy_summary["lpips"],
                "mean_psnr_gain": policy_summary["psnr"] - reference_summary["psnr"],
                "mean_ssim_gain": policy_summary["ssim"] - reference_summary["ssim"],
                "mean_lpips_gain": reference_summary["lpips"] - policy_summary["lpips"],
                "unbounded_selected_frame": int(reference_worst["selected_memory_frame"]),
                "unbounded_context_slot": int(reference_worst.get("context_slot", -1)),
                "unbounded_selected_overlap": finite_float(reference_worst.get("selected_overlap")),
                "unbounded_selected_psnr": finite_float(reference_worst.get("decoded_memory_psnr")),
                "unbounded_selected_ssim": finite_float(reference_worst.get("decoded_memory_ssim")),
                "unbounded_selected_lpips": finite_float(reference_worst.get("decoded_memory_lpips")),
                "bounded_selected_frame": int(policy_worst["selected_memory_frame"]),
                "bounded_context_slot": int(policy_worst.get("context_slot", -1)),
                "bounded_selected_overlap": finite_float(policy_worst.get("selected_overlap")),
                "bounded_selected_psnr": finite_float(policy_worst.get("decoded_memory_psnr")),
                "bounded_selected_ssim": finite_float(policy_worst.get("decoded_memory_ssim")),
                "bounded_selected_lpips": finite_float(policy_worst.get("decoded_memory_lpips")),
            }
        )
    return rows


def actual_threshold_pass(row, min_psnr_gain, min_ssim_gain, min_lpips_gain):
    return (
        row["mean_psnr_gain"] >= float(min_psnr_gain)
        and row["mean_ssim_gain"] >= float(min_ssim_gain)
        and row["mean_lpips_gain"] >= float(min_lpips_gain)
    )


def actual_rank_key(row):
    return (
        -float(row["mean_lpips_gain"]),
        -float(row["mean_psnr_gain"]),
        -float(row["mean_ssim_gain"]),
        int(row["batch_idx"]),
        int(row["target_frame"]),
    )


def select_diverse(rows, max_examples, per_batch, min_target_gap, rank_key):
    selected = []
    by_batch = defaultdict(list)
    for row in sorted(rows, key=rank_key):
        batch = int(row["batch_idx"])
        if len(by_batch[batch]) >= int(per_batch):
            continue
        if any(
            abs(int(row["target_frame"]) - int(previous["target_frame"]))
            < int(min_target_gap)
            for previous in by_batch[batch]
        ):
            continue
        selected.append(row)
        by_batch[batch].append(row)
        if len(selected) >= int(max_examples):
            break
    return selected


def prediction_paths(run_dir, limit=None):
    return dict(list_prediction_videos(run_dir, limit=limit, require_prefix=False))


def read_frames_single_pass(video_path, indices):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for the CPU visualizer") from exc
    wanted = sorted(set(int(index) for index in indices if int(index) >= 0))
    if not wanted:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    output = {}
    wanted_set = set(wanted)
    last = wanted[-1]
    index = 0
    try:
        while index <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if index in wanted_set:
                output[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        cap.release()
    missing = sorted(wanted_set - set(output))
    if missing:
        raise RuntimeError(f"Missing frames {missing[:10]} from {video_path}")
    return output


def resize_like(image, reference):
    if image.shape[:2] == reference.shape[:2]:
        return image
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for resizing") from exc
    height, width = reference.shape[:2]
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def frame_psnr(prediction, target, cap=100.0):
    prediction = np.asarray(prediction, dtype=np.float64) / 255.0
    target = resize_like(np.asarray(target), prediction).astype(np.float64) / 255.0
    mse = float(np.mean(np.square(prediction - target)))
    if mse <= 1e-12:
        return float(cap)
    return min(float(cap), float(-10.0 * np.log10(mse)))


def frame_ssim(prediction, target):
    """Compute the standard Gaussian-window SSIM entirely on CPU."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for SSIM") from exc
    prediction = np.asarray(prediction, dtype=np.float64) / 255.0
    target = resize_like(np.asarray(target), prediction).astype(np.float64) / 255.0
    kernel = (11, 11)
    mu_pred = cv2.GaussianBlur(prediction, kernel, 1.5)
    mu_gt = cv2.GaussianBlur(target, kernel, 1.5)
    mu_pred_sq = np.square(mu_pred)
    mu_gt_sq = np.square(mu_gt)
    mu_cross = mu_pred * mu_gt
    sigma_pred = cv2.GaussianBlur(np.square(prediction), kernel, 1.5) - mu_pred_sq
    sigma_gt = cv2.GaussianBlur(np.square(target), kernel, 1.5) - mu_gt_sq
    sigma_cross = cv2.GaussianBlur(prediction * target, kernel, 1.5) - mu_cross
    numerator = (2 * mu_cross + 0.01**2) * (2 * sigma_cross + 0.03**2)
    denominator = (mu_pred_sq + mu_gt_sq + 0.01**2) * (
        sigma_pred + sigma_gt + 0.03**2
    )
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def common_item_summary(items, metrics):
    scored = [metrics[int(row["selected_memory_frame"])] for row in items]
    return {
        "items": len(scored),
        "psnr": mean(row["psnr"] for row in scored),
        "ssim": mean(row["ssim"] for row in scored),
    }


def worst_common_item(items, metrics):
    return min(
        items,
        key=lambda row: (
            metrics[int(row["selected_memory_frame"])]["psnr"],
            metrics[int(row["selected_memory_frame"])]["ssim"],
            int(row.get("context_slot", -1)),
        ),
    )


def build_common_source_candidates(
    reference_groups,
    policy_groups,
    reference_videos,
    data_dir,
    context_frames,
    initial_skip_frames,
    fps,
    late_start_sec,
    dataset_seed,
    batches,
):
    """Score both selectors using pixels from the same Unbounded rollout."""
    output = []
    for batch in batches:
        keys = [
            key
            for key in sorted(set(reference_groups) & set(policy_groups))
            if key[0] == batch
            and (key[1] - int(context_frames)) / float(fps) >= float(late_start_sec)
        ]
        if not keys:
            continue
        by_key = {}
        local_indices = set()
        for key in keys:
            reference_items = generated_rows(reference_groups[key], context_frames)
            policy_items = generated_rows(policy_groups[key], context_frames)
            if not reference_items or not policy_items:
                continue
            by_key[key] = (reference_items, policy_items)
            local_indices.update(int(row["selected_memory_frame"]) for row in reference_items)
            local_indices.update(int(row["selected_memory_frame"]) for row in policy_items)
        if not by_key:
            continue
        if batch not in reference_videos:
            raise FileNotFoundError(f"No Unbounded prediction video for batch {batch}")

        pred_frames = read_frames_single_pass(
            reference_videos[batch],
            [index - int(context_frames) for index in local_indices],
        )
        gt_path = resolve_dataset_video_for_batch(
            data_dir=data_dir,
            batch_idx=batch,
            seed=dataset_seed,
            split="test",
            wo_updown=False,
        )
        gt_frames = read_frames_single_pass(
            gt_path,
            [int(initial_skip_frames) + index for index in local_indices],
        )
        metrics = {}
        for local_index in local_indices:
            prediction = pred_frames[local_index - int(context_frames)]
            target = gt_frames[int(initial_skip_frames) + local_index]
            metrics[local_index] = {
                "psnr": frame_psnr(prediction, target),
                "ssim": frame_ssim(prediction, target),
            }

        for (batch_idx, target_frame), (reference_items, policy_items) in by_key.items():
            reference_summary = common_item_summary(reference_items, metrics)
            policy_summary = common_item_summary(policy_items, metrics)
            reference_worst = worst_common_item(reference_items, metrics)
            policy_worst = worst_common_item(policy_items, metrics)
            reference_frame = int(reference_worst["selected_memory_frame"])
            policy_frame = int(policy_worst["selected_memory_frame"])
            output.append(
                {
                    "batch_idx": batch_idx,
                    "target_frame": target_frame,
                    "horizon_sec": (target_frame - int(context_frames)) / float(fps),
                    "unbounded_generated_items": reference_summary["items"],
                    "bounded_generated_items": policy_summary["items"],
                    "unbounded_mean_psnr": reference_summary["psnr"],
                    "unbounded_mean_ssim": reference_summary["ssim"],
                    "bounded_mean_psnr": policy_summary["psnr"],
                    "bounded_mean_ssim": policy_summary["ssim"],
                    "mean_psnr_gain": policy_summary["psnr"] - reference_summary["psnr"],
                    "mean_ssim_gain": policy_summary["ssim"] - reference_summary["ssim"],
                    "unbounded_selected_frame": reference_frame,
                    "unbounded_context_slot": int(reference_worst.get("context_slot", -1)),
                    "unbounded_selected_overlap": finite_float(reference_worst.get("selected_overlap")),
                    "unbounded_selected_psnr": metrics[reference_frame]["psnr"],
                    "unbounded_selected_ssim": metrics[reference_frame]["ssim"],
                    "bounded_selected_frame": policy_frame,
                    "bounded_context_slot": int(policy_worst.get("context_slot", -1)),
                    "bounded_selected_overlap": finite_float(policy_worst.get("selected_overlap")),
                    "bounded_selected_psnr": metrics[policy_frame]["psnr"],
                    "bounded_selected_ssim": metrics[policy_frame]["ssim"],
                }
            )
        print(
            f"[common-source] batch={batch:02d} chunks={len(by_key)} "
            f"unique_frames={len(local_indices)}"
        )
    return output


def common_threshold_pass(row, min_psnr_gain, min_ssim_gain):
    return (
        row["mean_psnr_gain"] >= float(min_psnr_gain)
        and row["mean_ssim_gain"] >= float(min_ssim_gain)
    )


def common_rank_key(row):
    return (
        -float(row["mean_psnr_gain"]),
        -float(row["mean_ssim_gain"]),
        int(row["batch_idx"]),
        int(row["target_frame"]),
    )


def load_font(size, bold=False):
    names = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ]
        if bold
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
    )
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size=size)
    return ImageFont.load_default()


def fit_font(draw, text, max_width, start_size, bold=False, min_size=9):
    for size in range(int(start_size), int(min_size) - 1, -1):
        font = load_font(size, bold=bold)
        bounds = draw.textbbox((0, 0), str(text), font=font)
        if bounds[2] - bounds[0] <= int(max_width):
            return font
    return load_font(min_size, bold=bold)


def as_pil(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")


def make_tile(title, image, footer_lines, color):
    tile = Image.new("RGB", (TILE_WIDTH, TILE_HEIGHT), (248, 248, 249))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, TILE_WIDTH - 1, TILE_HEIGHT - 1), outline=color, width=5)
    draw.line((0, HEADER_HEIGHT, TILE_WIDTH, HEADER_HEIGHT), fill=(20, 20, 20), width=4)
    draw.text(
        (TILE_WIDTH // 2, HEADER_HEIGHT // 2),
        title,
        fill=(24, 27, 31),
        anchor="mm",
        font=fit_font(draw, title, TILE_WIDTH - 18, 16, bold=True),
    )
    fitted = ImageOps.fit(
        as_pil(image),
        (TILE_WIDTH - 10, IMAGE_HEIGHT),
        method=Image.Resampling.LANCZOS,
    )
    tile.paste(fitted, (5, HEADER_HEIGHT))
    footer_y = HEADER_HEIGHT + IMAGE_HEIGHT + 7
    for line in footer_lines:
        draw.text(
            (TILE_WIDTH // 2, footer_y),
            line,
            fill=(36, 39, 44),
            anchor="ma",
            font=fit_font(draw, line, TILE_WIDTH - 18, 12),
        )
        footer_y += 18
    return tile


def format_overlap(value):
    value = finite_float(value)
    return "n/a" if value is None else f"{value:.3f}"


def load_case_assets(
    case,
    mode,
    reference_videos,
    policy_videos,
    data_dir,
    context_frames,
    initial_skip_frames,
    dataset_seed,
):
    batch = int(case["batch_idx"])
    target = int(case["target_frame"])
    reference_frame = int(case["unbounded_selected_frame"])
    policy_frame = int(case["bounded_selected_frame"])
    for name, value in (
        ("unbounded", reference_frame),
        ("bounded", policy_frame),
    ):
        if value < int(context_frames):
            raise RuntimeError(
                f"Selected {name} frame {value} is initial context; expected generated-only"
            )
    if batch not in reference_videos:
        raise FileNotFoundError(f"No Unbounded prediction video for batch {batch}")
    reference_indices = [reference_frame - int(context_frames)]
    if mode == "common_source":
        reference_indices.append(policy_frame - int(context_frames))
    reference_images = read_frames_single_pass(reference_videos[batch], reference_indices)
    if mode == "actual_history":
        if batch not in policy_videos:
            raise FileNotFoundError(f"No bounded prediction video for batch {batch}")
        policy_image = read_frames_single_pass(
            policy_videos[batch], [policy_frame - int(context_frames)]
        )[policy_frame - int(context_frames)]
    else:
        policy_image = reference_images[policy_frame - int(context_frames)]

    gt_path = resolve_dataset_video_for_batch(
        data_dir=data_dir,
        batch_idx=batch,
        seed=dataset_seed,
        split="test",
        wo_updown=False,
    )
    source_indices = [
        int(initial_skip_frames) + target,
        int(initial_skip_frames) + reference_frame,
        int(initial_skip_frames) + policy_frame,
    ]
    gt = read_frames_single_pass(gt_path, source_indices)
    return {
        "target_gt": gt[int(initial_skip_frames) + target],
        "unbounded": reference_images[reference_frame - int(context_frames)],
        "unbounded_gt": gt[int(initial_skip_frames) + reference_frame],
        "bounded": policy_image,
        "bounded_gt": gt[int(initial_skip_frames) + policy_frame],
        "gt_path": gt_path,
    }


def case_tiles(case, assets, policy_display, mode, initial_skip_frames):
    target = int(case["target_frame"])
    reference_frame = int(case["unbounded_selected_frame"])
    policy_frame = int(case["bounded_selected_frame"])
    common_suffix = " (U source)" if mode == "common_source" else ""
    reference_metric_line = (
        f"PSNR {case['unbounded_selected_psnr']:.2f} | "
        f"SSIM {case['unbounded_selected_ssim']:.3f}"
    )
    policy_metric_line = (
        f"PSNR {case['bounded_selected_psnr']:.2f} | "
        f"SSIM {case['bounded_selected_ssim']:.3f}"
    )
    if mode == "actual_history":
        reference_metric_line += f" | LPIPS {case['unbounded_selected_lpips']:.3f}"
        policy_metric_line += f" | LPIPS {case['bounded_selected_lpips']:.3f}"
    reference_footer = [
        f"worst of {case['unbounded_generated_items']} generated refs; m={reference_frame}",
        f"age={target-reference_frame} slot={case['unbounded_context_slot']} IoU={format_overlap(case['unbounded_selected_overlap'])}",
        reference_metric_line,
    ]
    policy_footer = [
        f"worst of {case['bounded_generated_items']} generated refs; m={policy_frame}",
        f"age={target-policy_frame} slot={case['bounded_context_slot']} IoU={format_overlap(case['bounded_selected_overlap'])}",
        policy_metric_line,
    ]
    return [
        make_tile(
            "Target ground truth",
            assets["target_gt"],
            [f"query local frame {target}", f"dataset frame {initial_skip_frames + target}"],
            COLORS["target"],
        ),
        make_tile(
            f"Unbounded selected{common_suffix}",
            assets["unbounded"],
            reference_footer,
            COLORS["unbounded"],
        ),
        make_tile(
            "GT at Unbounded index",
            assets["unbounded_gt"],
            [f"exact local frame {reference_frame}", f"dataset frame {initial_skip_frames + reference_frame}"],
            COLORS["ground_truth"],
        ),
        make_tile(
            f"{policy_display} selected{common_suffix}",
            assets["bounded"],
            policy_footer,
            COLORS["bounded"],
        ),
        make_tile(
            f"GT at {policy_display} index",
            assets["bounded_gt"],
            [f"exact local frame {policy_frame}", f"dataset frame {initial_skip_frames + policy_frame}"],
            COLORS["ground_truth"],
        ),
    ]


def render_gallery(
    cases,
    mode,
    output_dir,
    policy_display,
    reference_videos,
    policy_videos,
    data_dir,
    context_frames,
    initial_skip_frames,
    dataset_seed,
):
    mode_dir = Path(output_dir) / "figures" / mode
    cases_dir = mode_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    title_height = 102
    canvas = Image.new(
        "RGB",
        (TILE_WIDTH * 5, title_height + TILE_HEIGHT * len(cases)),
        (233, 235, 238),
    )
    draw = ImageDraw.Draw(canvas)
    title = (
        "Extreme retrieved-memory quality failures"
        if mode == "actual_history"
        else "Extreme common-source retrieval failures"
    )
    subtitle = (
        "Each selector uses its own matched rollout; events rank the mean quality gap across all generated retrieved memories"
        if mode == "actual_history"
        else "Both selectors read pixels from the Unbounded rollout; PSNR/SSIM use exact-index ground truth"
    )
    draw.text(
        (canvas.width // 2, 29),
        title,
        fill=(18, 23, 29),
        anchor="mm",
        font=load_font(28, bold=True),
    )
    draw.text(
        (canvas.width // 2, 69),
        subtitle,
        fill=(56, 61, 69),
        anchor="mm",
        font=load_font(14),
    )

    for index, case in enumerate(cases):
        assets = load_case_assets(
            case,
            mode,
            reference_videos,
            policy_videos,
            data_dir,
            context_frames,
            initial_skip_frames,
            dataset_seed,
        )
        tiles = case_tiles(
            case,
            assets,
            policy_display,
            mode,
            initial_skip_frames,
        )
        y = title_height + index * TILE_HEIGHT
        strip = Image.new("RGB", (TILE_WIDTH * 5, TILE_HEIGHT), (233, 235, 238))
        for column, tile in enumerate(tiles):
            canvas.paste(tile, (column * TILE_WIDTH, y))
            strip.paste(tile, (column * TILE_WIDTH, 0))
        stem = (
            f"case_{index:02d}_batch{int(case['batch_idx']):02d}_"
            f"target{int(case['target_frame']):04d}"
        )
        strip.save(cases_dir / f"{stem}.png")

    png_path = mode_dir / f"worldmem_{mode}_retrieval_failures.png"
    pdf_path = mode_dir / f"worldmem_{mode}_retrieval_failures.pdf"
    canvas.save(png_path)
    canvas.convert("RGB").save(pdf_path, "PDF", resolution=150.0)
    return png_path, pdf_path


def label_for_policy(metadata, batches, override):
    if override:
        return override
    row = metadata[min(batches)]
    policy = str(row.get("memory_policy", "Bounded")).replace("_", " ").title()
    budget = row.get("memory_budget")
    return f"{policy} B{int(budget)}" if budget not in (None, "", "None") else policy


def add_threshold_flags(rows, mode, args):
    for row in rows:
        if mode == "actual_history":
            row["meets_display_thresholds"] = actual_threshold_pass(
                row,
                args.min_actual_psnr_gain,
                args.min_actual_ssim_gain,
                args.min_actual_lpips_gain,
            )
        else:
            row["meets_display_thresholds"] = common_threshold_pass(
                row,
                args.min_common_psnr_gain,
                args.min_common_ssim_gain,
            )


def choose_cases(rows, mode, args):
    passing = [row for row in rows if row["meets_display_thresholds"]]
    if args.strict_thresholds and not passing:
        raise RuntimeError(f"No {mode} cases passed the display thresholds")
    pool = passing or rows
    rank_key = actual_rank_key if mode == "actual_history" else common_rank_key
    return select_diverse(
        pool,
        max_examples=args.max_examples,
        per_batch=args.per_batch,
        min_target_gap=args.min_target_gap,
        rank_key=rank_key,
    )


def write_report(path, args, policy_display, results):
    lines = [
        "# WorldMem Retrieval-Failure Gallery",
        "",
        "## Protocol",
        "",
        f"- Comparison: `Unbounded` versus `{policy_display}`.",
        f"- Window: `{args.late_start_sec:g}-60` seconds.",
        "- One event per trajectory by default; repeated retrievals remain repeated in event means.",
        "- Each visual tile is the worst generated-memory item among the eight retrieved slots.",
        "- Events are ranked by the mean quality difference across every generated memory retrieved at that chunk, not by the displayed item alone.",
        "- Actual-history metrics come from VAE-decoded latent trace values.",
        "- Common-source metrics are CPU MP4-versus-exact-index-GT PSNR/SSIM.",
        "",
        "## Outputs",
        "",
    ]
    for mode, cases in results.items():
        lines.extend(
            [
                f"### {mode.replace('_', ' ').title()}",
                "",
                f"- Scored events: `{cases['total']}`.",
                f"- Threshold-passing events: `{cases['passing']}`.",
                f"- Displayed cases: `{len(cases['selected'])}`.",
                f"- Figure: `figures/{mode}/worldmem_{mode}_retrieval_failures.png`.",
                f"- Individual strips: `figures/{mode}/cases/case_*.png`.",
                f"- Candidate table: `tables/{mode}_all_candidates.csv`.",
                f"- Selected table: `tables/{mode}_selected_cases.csv`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation Guardrail",
            "",
            "Actual-history examples show the conditioning evidence each policy really used, but combine retrieval decisions with earlier differences in rollout quality. Common-source examples hold source pixels fixed to Unbounded and isolate selected frame identity. These are deliberately selected extremes and must be presented beside the aggregate 15-trajectory statistics, not as frequency estimates.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality_root", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--reference_run", required=True)
    parser.add_argument("--policy_run", required=True)
    parser.add_argument("--policy_display", default="")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--modes", default="actual_history,common_source")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--context_frames", type=int, default=600)
    parser.add_argument("--initial_skip_frames", type=int, default=100)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--late_start_sec", type=float, default=45.0)
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--max_examples", type=int, default=5)
    parser.add_argument("--per_batch", type=int, default=1)
    parser.add_argument("--min_target_gap", type=int, default=80)
    parser.add_argument("--min_actual_psnr_gain", type=float, default=3.0)
    parser.add_argument("--min_actual_ssim_gain", type=float, default=0.05)
    parser.add_argument("--min_actual_lpips_gain", type=float, default=0.15)
    parser.add_argument("--min_common_psnr_gain", type=float, default=2.0)
    parser.add_argument("--min_common_ssim_gain", type=float, default=0.03)
    parser.add_argument("--strict_thresholds", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    invalid = sorted(set(modes) - {"actual_history", "common_source"})
    if invalid:
        raise ValueError(f"Unknown modes: {invalid}")
    reference_dir = args.quality_root / args.reference_run
    policy_dir = args.quality_root / args.policy_run
    for path in (reference_dir, policy_dir, args.data_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    reference_groups, reference_metadata = load_quality_trace(find_trace(reference_dir))
    policy_groups, policy_metadata = load_quality_trace(find_trace(policy_dir))
    matched_batches = sorted(
        {
            batch
            for batch, _target in set(reference_groups) & set(policy_groups)
        }
    )
    if args.limit is not None:
        required = list(range(int(args.limit)))
        missing = sorted(set(required) - set(matched_batches))
        if missing:
            raise RuntimeError(f"Missing matched trace batches: {missing}")
        matched_batches = required
    if not matched_batches:
        raise RuntimeError("No matched trace batches")
    validate_matched_runs(reference_metadata, policy_metadata, matched_batches)
    allowed = set(matched_batches)
    reference_groups = {
        key: rows for key, rows in reference_groups.items() if key[0] in allowed
    }
    policy_groups = {
        key: rows for key, rows in policy_groups.items() if key[0] in allowed
    }
    policy_display = label_for_policy(policy_metadata, matched_batches, args.policy_display)
    reference_videos = prediction_paths(reference_dir, limit=args.limit)
    policy_videos = prediction_paths(policy_dir, limit=args.limit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    if "actual_history" in modes:
        rows = build_actual_candidates(
            reference_groups,
            policy_groups,
            args.context_frames,
            args.fps,
            args.late_start_sec,
        )
        add_threshold_flags(rows, "actual_history", args)
        selected = choose_cases(rows, "actual_history", args)
        write_csv(args.output_dir / "tables" / "actual_history_all_candidates.csv", rows)
        write_csv(args.output_dir / "tables" / "actual_history_selected_cases.csv", selected)
        render_gallery(
            selected,
            "actual_history",
            args.output_dir,
            policy_display,
            reference_videos,
            policy_videos,
            args.data_dir,
            args.context_frames,
            args.initial_skip_frames,
            args.dataset_seed,
        )
        results["actual_history"] = {
            "total": len(rows),
            "passing": sum(bool(row["meets_display_thresholds"]) for row in rows),
            "selected": selected,
        }

    if "common_source" in modes:
        rows = build_common_source_candidates(
            reference_groups,
            policy_groups,
            reference_videos,
            args.data_dir,
            args.context_frames,
            args.initial_skip_frames,
            args.fps,
            args.late_start_sec,
            args.dataset_seed,
            matched_batches,
        )
        add_threshold_flags(rows, "common_source", args)
        selected = choose_cases(rows, "common_source", args)
        write_csv(args.output_dir / "tables" / "common_source_all_candidates.csv", rows)
        write_csv(args.output_dir / "tables" / "common_source_selected_cases.csv", selected)
        render_gallery(
            selected,
            "common_source",
            args.output_dir,
            policy_display,
            reference_videos,
            policy_videos,
            args.data_dir,
            args.context_frames,
            args.initial_skip_frames,
            args.dataset_seed,
        )
        results["common_source"] = {
            "total": len(rows),
            "passing": sum(bool(row["meets_display_thresholds"]) for row in rows),
            "selected": selected,
        }

    write_report(args.output_dir / "report.md", args, policy_display, results)
    for mode, result in results.items():
        print(
            f"[{mode}] events={result['total']} "
            f"passing={result['passing']} selected={len(result['selected'])}"
        )
    print(f"Wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
