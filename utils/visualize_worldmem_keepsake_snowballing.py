#!/usr/bin/env python3
"""Build a CPU-only four-column KEEPSAKE snowballing figure."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.visualize_worldmem_retrieval_failures import (  # noqa: E402
    find_trace,
    frame_psnr,
    frame_ssim,
    load_quality_trace,
    prediction_paths,
    read_frames_single_pass,
    resize_like,
    validate_matched_runs,
)
from utils.worldmem_eval_common import resolve_dataset_video_for_batch  # noqa: E402


TILE_WIDTH = 320
IMAGE_HEIGHT = 196
HEADER_HEIGHT = 44
FOOTER_HEIGHT = 58
TILE_HEIGHT = HEADER_HEIGHT + IMAGE_HEIGHT + FOOTER_HEIGHT
TITLE_HEIGHT = 108
COLORS = {
    "gt": (54, 102, 164),
    "unbounded": (194, 61, 56),
    "fifo": (219, 125, 29),
    "keepsake": (39, 145, 83),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--unbounded-run", required=True)
    parser.add_argument("--fifo-run", required=True)
    parser.add_argument("--keepsake-run", required=True)
    parser.add_argument("--cases-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-frames", type=int, default=600)
    parser.add_argument("--initial-skip-frames", type=int, default=100)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--max-examples", type=int, default=5)
    return parser.parse_args()


def load_font(size: int, bold: bool = False):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def fit_font(draw, text, max_width, start_size, bold=False, min_size=9):
    for size in range(start_size, min_size - 1, -1):
        font = load_font(size, bold=bold)
        if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
            return font
    return load_font(min_size, bold=bold)


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
        font=fit_font(draw, title, TILE_WIDTH - 20, 17, bold=True),
    )
    fitted = ImageOps.fit(
        Image.fromarray(image),
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


def load_cases(path: Path, max_examples: int) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Selected-case table not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"batch_idx", "target_frame"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"Case table must contain {sorted(required)}: {path}")
    return rows[: int(max_examples)]


def validate_run_identity(run_dirs, batches):
    metadata = []
    for run_dir in run_dirs:
        _groups, run_metadata = load_quality_trace(find_trace(run_dir))
        metadata.append(run_metadata)
    for current in metadata[1:]:
        validate_matched_runs(metadata[0], current, batches)


def metric_footer(prediction, ground_truth):
    prediction = resize_like(prediction, ground_truth)
    return (
        f"PSNR {frame_psnr(prediction, ground_truth):.2f} dB | "
        f"SSIM {frame_ssim(prediction, ground_truth):.3f}"
    )


def render_case_tiles(case, images, context_frames, fps):
    batch = int(case["batch_idx"])
    target = int(case["target_frame"])
    horizon = (target - int(context_frames)) / float(fps)
    identity = f"batch {batch:02d} | future {horizon:.1f}s | frame {target}"
    gt = images["gt"]
    return [
        make_tile("Ground truth", gt, [identity, "exact-index dataset frame"], COLORS["gt"]),
        make_tile(
            "Unbounded",
            images["unbounded"],
            [identity, metric_footer(images["unbounded"], gt)],
            COLORS["unbounded"],
        ),
        make_tile(
            "FIFO",
            images["fifo"],
            [identity, metric_footer(images["fifo"], gt)],
            COLORS["fifo"],
        ),
        make_tile(
            "KEEPSAKE (ours)",
            images["keepsake"],
            [identity, metric_footer(images["keepsake"], gt)],
            COLORS["keepsake"],
        ),
    ]


def load_case_images(case, videos, args):
    batch = int(case["batch_idx"])
    target = int(case["target_frame"])
    prediction_index = target - int(args.context_frames)
    if prediction_index < 0:
        raise ValueError(f"Target frame {target} is not in the generated horizon")
    images = {}
    for label, paths in videos.items():
        if batch not in paths:
            raise FileNotFoundError(f"No {label} prediction video for batch {batch}")
        images[label] = read_frames_single_pass(paths[batch], [prediction_index])[
            prediction_index
        ]
    gt_path = resolve_dataset_video_for_batch(
        data_dir=args.data_dir,
        batch_idx=batch,
        seed=args.dataset_seed,
        split="test",
        wo_updown=False,
    )
    gt_index = int(args.initial_skip_frames) + target
    images["gt"] = read_frames_single_pass(gt_path, [gt_index])[gt_index]
    return images


def write_metrics(path, records):
    fields = (
        "batch_idx",
        "target_frame",
        "future_seconds",
        "unbounded_psnr",
        "unbounded_ssim",
        "fifo_psnr",
        "fifo_ssim",
        "keepsake_psnr",
        "keepsake_ssim",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases_csv, args.max_examples)
    run_dirs = {
        "unbounded": args.quality_root / args.unbounded_run,
        "fifo": args.quality_root / args.fifo_run,
        "keepsake": args.quality_root / args.keepsake_run,
    }
    for path in (*run_dirs.values(), args.data_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    batches = sorted({int(case["batch_idx"]) for case in cases})
    validate_run_identity(list(run_dirs.values()), batches)
    videos = {
        label: prediction_paths(run_dir, limit=args.limit)
        for label, run_dir in run_dirs.items()
    }

    canvas = Image.new(
        "RGB",
        (TILE_WIDTH * 4, TITLE_HEIGHT + TILE_HEIGHT * len(cases)),
        (233, 235, 238),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (canvas.width // 2, 30),
        "Selective memory limits autoregressive error snowballing",
        fill=(18, 23, 29),
        anchor="mm",
        font=load_font(27, bold=True),
    )
    draw.text(
        (canvas.width // 2, 71),
        "Matched trajectories and frame indices; every prediction uses the same exact-index ground truth",
        fill=(56, 61, 69),
        anchor="mm",
        font=load_font(14),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = args.output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, case in enumerate(cases):
        images = load_case_images(case, videos, args)
        tiles = render_case_tiles(case, images, args.context_frames, args.fps)
        strip = Image.new("RGB", (TILE_WIDTH * 4, TILE_HEIGHT), (233, 235, 238))
        y = TITLE_HEIGHT + index * TILE_HEIGHT
        for column, tile in enumerate(tiles):
            canvas.paste(tile, (column * TILE_WIDTH, y))
            strip.paste(tile, (column * TILE_WIDTH, 0))
        batch = int(case["batch_idx"])
        target = int(case["target_frame"])
        strip.save(cases_dir / f"case_{index:02d}_batch{batch:02d}_target{target:04d}.png")
        gt = images["gt"]
        record = {
            "batch_idx": batch,
            "target_frame": target,
            "future_seconds": (target - args.context_frames) / args.fps,
        }
        for label in ("unbounded", "fifo", "keepsake"):
            prediction = resize_like(images[label], gt)
            record[f"{label}_psnr"] = frame_psnr(prediction, gt)
            record[f"{label}_ssim"] = frame_ssim(prediction, gt)
        records.append(record)

    stem = args.output_dir / "worldmem_keepsake_snowballing_4col"
    canvas.save(stem.with_suffix(".png"))
    canvas.convert("RGB").save(stem.with_suffix(".pdf"), "PDF", resolution=150.0)
    write_metrics(args.output_dir / "selected_case_metrics.csv", records)
    print(f"Wrote: {stem.with_suffix('.png')}")
    print(f"Wrote: {stem.with_suffix('.pdf')}")
    print(f"Wrote: {args.output_dir / 'selected_case_metrics.csv'}")


if __name__ == "__main__":
    main()
