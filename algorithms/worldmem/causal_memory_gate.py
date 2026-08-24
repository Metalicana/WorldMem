import json
import math
from pathlib import Path

import numpy as np


CALIBRATION_VERSION = 1


def cosine_similarity(first, second):
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def rotation_distance_rad(first, second):
    relative = np.asarray(first[:3, :3], dtype=np.float64).T @ np.asarray(
        second[:3, :3], dtype=np.float64
    )
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def pose_components(c2ws, target_frame, context_frame):
    target = np.asarray(c2ws[int(target_frame)], dtype=np.float64)
    context = np.asarray(c2ws[int(context_frame)], dtype=np.float64)
    return {
        "translation": float(np.linalg.norm(target[:3, 3] - context[:3, 3])),
        "rotation_rad": rotation_distance_rad(context, target),
    }


def calibrated_pose_distance(components, pose_scales):
    translation_scale = max(float(pose_scales["translation"]), 1e-12)
    rotation_scale = max(float(pose_scales["rotation_rad"]), 1e-12)
    return float(
        float(components["translation"]) / translation_scale
        + float(components["rotation_rad"]) / rotation_scale
    )


def expected_similarity(expectation, distance):
    edges = np.asarray(expectation["edges"], dtype=np.float64)
    means = np.asarray(expectation["means"], dtype=np.float64)
    if means.size != edges.size + 1:
        raise ValueError("causal gate expectation must have len(edges) + 1 means")
    bin_index = int(np.searchsorted(edges, float(distance), side="right"))
    return float(means[bin_index])


def normalized_context_weights(overlaps):
    overlaps = np.asarray(overlaps, dtype=np.float64)
    if overlaps.ndim != 1 or overlaps.size == 0:
        raise ValueError("causal gate needs at least one context overlap")
    positive = np.maximum(np.where(np.isfinite(overlaps), overlaps, 0.0), 0.0)
    total = float(positive.sum())
    if total <= 1e-12:
        return np.full(overlaps.shape, 1.0 / overlaps.size, dtype=np.float64)
    return positive / total


def validate_calibration(calibration, require_approved=True):
    if int(calibration.get("version", -1)) != CALIBRATION_VERSION:
        raise ValueError(
            f"unsupported causal gate calibration version: "
            f"{calibration.get('version')!r}"
        )
    if calibration.get("feature_backend") != "latent":
        raise ValueError("WorldMem causal gate currently requires latent calibration")
    threshold = float(calibration["threshold"])
    if not math.isfinite(threshold):
        raise ValueError("causal gate threshold must be finite")
    pose_scales = calibration["pose_scales"]
    if float(pose_scales["translation"]) <= 0:
        raise ValueError("causal gate translation scale must be positive")
    if float(pose_scales["rotation_rad"]) <= 0:
        raise ValueError("causal gate rotation scale must be positive")
    expected_similarity(calibration["expectation"], 0.0)
    if require_approved and not bool(calibration.get("deployment_approved", False)):
        raise ValueError(
            "causal gate calibration is not approved for deployment; "
            "set causal_gate_require_approved=false only for a diagnostic run"
        )
    return calibration


def load_calibration(path, require_approved=True):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"causal gate calibration not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)
    return validate_calibration(calibration, require_approved=require_approved)


def causal_consistency_score(
    target_feature,
    context_features,
    c2ws,
    target_frame,
    context_frames,
    overlaps,
    calibration,
):
    context_frames = [int(frame) for frame in context_frames]
    if len(context_features) != len(context_frames):
        raise ValueError("context feature and frame counts must match")
    if len(overlaps) != len(context_frames):
        raise ValueError("context overlap and frame counts must match")
    weights = normalized_context_weights(overlaps)
    parent_rows = []
    for context_frame, context_feature, weight, overlap in zip(
        context_frames, context_features, weights, overlaps
    ):
        components = pose_components(c2ws, target_frame, context_frame)
        distance = calibrated_pose_distance(components, calibration["pose_scales"])
        similarity = cosine_similarity(target_feature, context_feature)
        expected = expected_similarity(calibration["expectation"], distance)
        parent_rows.append(
            {
                "frame": int(context_frame),
                "weight": float(weight),
                "overlap": None if overlap is None else float(overlap),
                "generated_similarity": similarity,
                "translation": components["translation"],
                "rotation_rad": components["rotation_rad"],
                "pose_distance": distance,
                "expected_similarity": expected,
                "residual": similarity - expected,
            }
        )
    score = float(
        sum(row["weight"] * row["residual"] for row in parent_rows)
    )
    threshold = float(calibration["threshold"])
    return {
        "score": score,
        "threshold": threshold,
        "admitted": bool(score >= threshold),
        "parents": parent_rows,
        "weight_source": (
            "positive_overlap" if any(max(float(v or 0.0), 0.0) > 0 for v in overlaps)
            else "uniform"
        ),
    }
