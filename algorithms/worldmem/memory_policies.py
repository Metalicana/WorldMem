import math
from collections import OrderedDict

import numpy as np


SUPPORTED_MEMORY_POLICIES = (
    "unbounded",
    "random_cap",
    "fifo",
    "rarity_irreplaceability",
    "slam_covisibility",
    "kcenter_coreset",
    "mce",
    "causal_consistency_coverage_ri",
    "coverage_hysteresis",
)
BUDGETED_MEMORY_POLICIES = (
    "random_cap",
    "fifo",
    "rarity_irreplaceability",
    "slam_covisibility",
    "kcenter_coreset",
    "mce",
    "causal_consistency_coverage_ri",
    "coverage_hysteresis",
)


class FrameMemoryBuffer:
    def __init__(
        self,
        policy="unbounded",
        budget=None,
        pinned_frames=None,
        random_seed=0,
    ):
        if policy not in SUPPORTED_MEMORY_POLICIES:
            raise ValueError(
                f"Unsupported memory policy '{policy}'. "
                f"Expected one of {SUPPORTED_MEMORY_POLICIES}."
            )
        if policy in BUDGETED_MEMORY_POLICIES and budget is None:
            raise ValueError(f"{policy} memory policy requires an explicit memory budget")
        if budget is not None and budget <= 0:
            raise ValueError("memory budget must be positive when provided")

        self.policy = policy
        self.budget = budget
        self._frames = OrderedDict()
        self._stats = {}
        self._next_order = 0
        self._pinned_frames = set(pinned_frames or [])
        self._rng = np.random.default_rng(int(random_seed))

    def add(self, frame_idx, evict=True, eviction_scores=None, protected_frames=None):
        frame_idx = int(frame_idx)
        if frame_idx not in self._frames:
            self._stats[frame_idx] = {
                "insert_order": self._next_order,
                "selected_count": 0,
                "selection_overlap_sum": 0.0,
                "best_selection_overlap": 0.0,
                "score": 0.0,
            }
            self._next_order += 1
        self._frames[frame_idx] = None
        if eviction_scores:
            self.set_scores(eviction_scores)
        if evict:
            return self.evict_to_budget(protected_frames=protected_frames)
        return []

    def update(self, frame_indices, eviction_scores=None, protected_frames=None):
        evicted = []
        for frame_idx in frame_indices:
            evicted.extend(self.add(frame_idx, evict=False))
        if eviction_scores:
            self.set_scores(eviction_scores)
        evicted.extend(self.evict_to_budget(protected_frames=protected_frames))
        return evicted

    def set_scores(self, scores):
        for frame_idx, score in scores.items():
            frame_idx = int(frame_idx)
            if frame_idx in self._stats:
                self._stats[frame_idx]["score"] = float(score)

    def record_selection(self, frame_idx, overlap):
        frame_idx = int(frame_idx)
        if frame_idx not in self._stats:
            return
        overlap = max(float(overlap or 0.0), 0.0)
        stats = self._stats[frame_idx]
        stats["selected_count"] += 1
        stats["selection_overlap_sum"] += overlap
        stats["best_selection_overlap"] = max(stats["best_selection_overlap"], overlap)

    def evict_to_budget(self, protected_frames=None):
        if self.budget is None or self.policy == "unbounded":
            return []

        protected_frames = set(protected_frames or []) | self._pinned_frames
        evicted = []
        while len(self._frames) > self.budget:
            evictable = [
                frame_idx
                for frame_idx in self._frames.keys()
                if frame_idx not in protected_frames
            ]
            if not evictable:
                break

            if self.policy == "fifo":
                evicted_frame_idx = evictable[0]
            elif self.policy == "random_cap":
                evicted_frame_idx = int(self._rng.choice(evictable))
            else:
                evicted_frame_idx = min(
                    evictable,
                    key=lambda idx: (
                        self._stats[idx].get("score", 0.0),
                        (
                            -self._stats[idx]["insert_order"]
                            if self.policy == "coverage_hysteresis"
                            else self._stats[idx]["insert_order"]
                        ),
                    ),
                )

            self._frames.pop(evicted_frame_idx, None)
            self._stats.pop(evicted_frame_idx, None)
            evicted.append(evicted_frame_idx)
        return evicted

    def candidates(self, exclude_frames=None):
        exclude_frames = set(exclude_frames or [])
        return [frame_idx for frame_idx in self._frames.keys() if frame_idx not in exclude_frames]

    def selected_count(self, frame_idx):
        return self._stats.get(int(frame_idx), {}).get("selected_count", 0)

    def frame_stats(self, frame_idx):
        return dict(self._stats.get(int(frame_idx), {}))

    def __len__(self):
        return len(self._frames)


def rotation_distance(rotation_a, rotation_b):
    relative = rotation_a.T @ rotation_b
    cosine = (np.trace(relative) - 1.0) / 2.0
    cosine = np.clip(cosine, -1.0, 1.0)
    return math.acos(cosine) / math.pi


def pose_distances(c2ws, frame_indices, target_indices, rotation_weight=2.0):
    frame_indices = list(frame_indices)
    target_indices = list(target_indices)
    if not frame_indices or not target_indices:
        return np.zeros((len(frame_indices), len(target_indices)), dtype=np.float64)

    frame_positions = c2ws[frame_indices, :3, 3]
    target_positions = c2ws[target_indices, :3, 3]
    position_dists = np.linalg.norm(
        frame_positions[:, None, :] - target_positions[None, :, :],
        axis=-1,
    )
    nonzero = position_dists[position_dists > 1e-8]
    position_scale = float(np.median(nonzero)) if nonzero.size else 1.0
    position_scale = max(position_scale, 1e-6)
    position_dists = position_dists / position_scale

    rotation_dists = np.zeros_like(position_dists)
    for row, frame_idx in enumerate(frame_indices):
        rotation_a = c2ws[frame_idx, :3, :3]
        for col, target_idx in enumerate(target_indices):
            rotation_b = c2ws[target_idx, :3, :3]
            rotation_dists[row, col] = rotation_distance(rotation_a, rotation_b)

    return position_dists + rotation_weight * rotation_dists


def camera_trajectory_similarity(
    c2ws,
    query_frame_indices,
    memory_frame_indices,
    fov_half_h=52.5,
    fov_half_v=37.5,
    radius=30.0,
):
    """Camera-only view affinity using WorldMem's +z-forward convention."""
    query_frame_indices = list(query_frame_indices)
    memory_frame_indices = list(memory_frame_indices)
    if not query_frame_indices or not memory_frame_indices:
        return np.zeros(
            (len(query_frame_indices), len(memory_frame_indices)),
            dtype=np.float64,
        )
    if radius <= 0:
        raise ValueError("radius must be positive")

    query = c2ws[query_frame_indices]
    memory = c2ws[memory_frame_indices]
    position_distance = np.linalg.norm(
        query[:, None, :3, 3] - memory[None, :, :3, 3],
        axis=-1,
    )
    position_similarity = np.clip(
        1.0 - position_distance / (2.0 * float(radius)),
        0.0,
        1.0,
    )

    query_forward = query[:, :3, 2]
    memory_forward = memory[:, :3, 2]
    query_yaw = np.arctan2(query_forward[:, 0], query_forward[:, 2])
    memory_yaw = np.arctan2(memory_forward[:, 0], memory_forward[:, 2])
    yaw_distance = np.abs(query_yaw[:, None] - memory_yaw[None, :])
    yaw_distance = np.minimum(yaw_distance, 2.0 * np.pi - yaw_distance)

    query_pitch = np.arctan2(
        query_forward[:, 1],
        np.linalg.norm(query_forward[:, [0, 2]], axis=1),
    )
    memory_pitch = np.arctan2(
        memory_forward[:, 1],
        np.linalg.norm(memory_forward[:, [0, 2]], axis=1),
    )
    pitch_distance = np.abs(query_pitch[:, None] - memory_pitch[None, :])

    horizontal_similarity = np.clip(
        1.0 - yaw_distance / (2.0 * np.deg2rad(float(fov_half_h))),
        0.0,
        1.0,
    )
    vertical_similarity = np.clip(
        1.0 - pitch_distance / (2.0 * np.deg2rad(float(fov_half_v))),
        0.0,
        1.0,
    )
    return position_similarity * horizontal_similarity * vertical_similarity


def select_coverage_hysteresis_admissions(
    existing_frame_indices,
    candidate_frame_indices,
    c2ws,
    view_similarity_threshold=0.90,
    fov_half_h=52.5,
    fov_half_v=37.5,
    radius=30.0,
    return_details=False,
):
    """Admit only candidates that add a view absent from persistent memory."""
    existing = list(dict.fromkeys(int(idx) for idx in existing_frame_indices))
    candidates = list(dict.fromkeys(int(idx) for idx in candidate_frame_indices))
    threshold = float(view_similarity_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("view_similarity_threshold must be in [0, 1]")

    references = list(existing)
    reference_set = set(references)
    admitted = []
    details = {}
    for frame_idx in candidates:
        if frame_idx in reference_set:
            details[frame_idx] = {
                "hysteresis_admitted": False,
                "hysteresis_reason": "already_stored",
                "hysteresis_max_view_similarity": 1.0,
                "hysteresis_nearest_reference_frame": int(frame_idx),
                "hysteresis_reference_count": len(references),
                "hysteresis_view_threshold": threshold,
            }
            continue

        if references:
            similarities = camera_trajectory_similarity(
                c2ws=c2ws,
                query_frame_indices=[frame_idx],
                memory_frame_indices=references,
                fov_half_h=fov_half_h,
                fov_half_v=fov_half_v,
                radius=radius,
            )[0]
            nearest_position = int(np.argmax(similarities))
            max_similarity = float(similarities[nearest_position])
            nearest_reference = int(references[nearest_position])
        else:
            max_similarity = 0.0
            nearest_reference = None

        should_admit = max_similarity < threshold
        details[frame_idx] = {
            "hysteresis_admitted": bool(should_admit),
            "hysteresis_reason": (
                "novel_view" if should_admit else "covered_by_incumbent"
            ),
            "hysteresis_max_view_similarity": max_similarity,
            "hysteresis_nearest_reference_frame": nearest_reference,
            "hysteresis_reference_count": len(references),
            "hysteresis_view_threshold": threshold,
        }
        if should_admit:
            admitted.append(frame_idx)
            references.append(frame_idx)
            reference_set.add(frame_idx)

    return (admitted, details) if return_details else admitted


def cosine_distances(features):
    features = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-12)
    similarities = np.clip(features @ features.T, -1.0, 1.0)
    return 1.0 - similarities


def pairwise_mean_abs_distances(features):
    features = np.asarray(features, dtype=np.float64)
    num_items = len(features)
    distances = np.zeros((num_items, num_items), dtype=np.float64)
    for index in range(num_items):
        if index + 1 >= num_items:
            continue
        row = np.mean(np.abs(features[index + 1 :] - features[index]), axis=1)
        distances[index, index + 1 :] = row
        distances[index + 1 :, index] = row
    return distances


def connected_components_from_threshold(pairwise_distances, threshold):
    num_items = pairwise_distances.shape[0]
    visited = np.zeros(num_items, dtype=bool)
    cluster_ids = np.full(num_items, -1, dtype=np.int64)
    clusters = []

    for start in range(num_items):
        if visited[start]:
            continue

        cluster_id = len(clusters)
        stack = [start]
        visited[start] = True
        members = []

        while stack:
            item = stack.pop()
            members.append(item)
            neighbors = np.flatnonzero(pairwise_distances[item] <= threshold)
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))

        for member in members:
            cluster_ids[member] = cluster_id
        clusters.append(members)

    return cluster_ids, clusters


def estimate_cluster_threshold(pairwise_distances, rarity_neighbors=3):
    """Median distance to each point's rarity_neighbors-th nearest neighbor.

    BUG HISTORY: this always used the 1st-nearest-neighbor distance
    (``np.partition(..., 0)[:, 0]``) with no way to ask for anything else --
    both ``compute_rarity_irreplaceability_scores`` (RI) and
    ``_historical_query_medoids`` (MCE's Q_hist) inherited that fixed k=1
    granularity. Ported from the same fix landed in MemCam's
    ``memory_policies.py`` after an MCE offline sweep there found clustering
    was insensitive to the (there, dead) ``rarity_neighbors`` parameter and a
    synthetic test showed 6 well-separated 10-item clusters still fragmenting
    into 37 near-singleton clusters regardless of the requested k.
    ``rarity_neighbors=1`` reproduces the old behavior exactly, so this is a
    strict generalization, not a change at any call site that never asked
    for anything else.
    """
    finite = pairwise_distances[np.isfinite(pairwise_distances)]
    if finite.size == 0:
        return 0.0

    # Use the median k-th-nearest-neighbor distance as the mode scale. Larger
    # k coarsens the threshold (merges more into one cluster); k=1 matches the
    # original single-nearest-neighbor behavior. Each row has exactly one
    # guaranteed-inf self-distance entry (diagonal), which sorts to the last
    # position after partitioning -- so the valid non-self neighbor range is
    # num_points - 2, not num_points - 1. Getting this off-by-one wrong lets
    # the index land on the self-inf entry for small candidate pools (this is
    # exactly what broke MemCam's own duplicate-view counterexample test at 3
    # candidates before they caught it -- ported here so WorldMem doesn't hit
    # the same edge case independently).
    num_points = pairwise_distances.shape[0]
    neighbor_index = max(0, min(int(rarity_neighbors) - 1, num_points - 2))
    nearest = np.partition(pairwise_distances, neighbor_index, axis=1)[:, neighbor_index]
    nearest = nearest[np.isfinite(nearest)]
    if nearest.size:
        return float(np.median(nearest))
    return float(np.median(finite))


def _feature_matrix(memory_frame_indices, features):
    missing = [idx for idx in memory_frame_indices if idx not in features]
    if missing:
        raise ValueError(f"Missing memory features for frames: {missing[:10]}")
    return np.stack([features[idx] for idx in memory_frame_indices])


def compute_rarity_irreplaceability_scores(
    memory_frame_indices,
    latent_features=None,
    pinned_frames=None,
    return_details=False,
    rarity_features=None,
    irreplaceability_features=None,
    irreplaceability_metric="cosine",
    rarity_neighbors=3,
):
    memory_frame_indices = list(memory_frame_indices)
    pinned_frames = set(pinned_frames or [])
    if not memory_frame_indices:
        return ({}, {}) if return_details else {}

    if rarity_features is None:
        rarity_features = latent_features
    if rarity_features is None:
        raise ValueError("RI requires latent_features or rarity_features")
    if irreplaceability_features is None:
        irreplaceability_features = rarity_features
    if irreplaceability_metric not in {"cosine", "mean_abs"}:
        raise ValueError("irreplaceability_metric must be 'cosine' or 'mean_abs'")

    rarity_matrix = _feature_matrix(memory_frame_indices, rarity_features)
    rarity_pairwise = cosine_distances(rarity_matrix)
    np.fill_diagonal(rarity_pairwise, np.inf)

    if len(memory_frame_indices) == 1:
        cluster_ids = np.zeros(1, dtype=np.int64)
        cluster_sizes = np.ones(1, dtype=np.float64)
        threshold = 0.0
        nearest_distances = np.ones(1, dtype=np.float64)
        nearest_indices = np.full(1, -1, dtype=np.int64)
    else:
        threshold = estimate_cluster_threshold(rarity_pairwise, rarity_neighbors)
        cluster_pairwise = rarity_pairwise.copy()
        np.fill_diagonal(cluster_pairwise, 0.0)
        cluster_ids, clusters = connected_components_from_threshold(
            cluster_pairwise,
            threshold=threshold,
        )
        cluster_sizes = np.array([len(clusters[cluster_id]) for cluster_id in cluster_ids])
        irreplaceability_matrix = _feature_matrix(
            memory_frame_indices,
            irreplaceability_features,
        )
        if irreplaceability_metric == "mean_abs":
            irreplaceability_pairwise = pairwise_mean_abs_distances(
                irreplaceability_matrix
            )
        else:
            irreplaceability_pairwise = cosine_distances(irreplaceability_matrix)
        np.fill_diagonal(irreplaceability_pairwise, np.inf)
        nearest_indices = np.argmin(irreplaceability_pairwise, axis=1)
        nearest_distances = irreplaceability_pairwise[
            np.arange(len(memory_frame_indices)), nearest_indices
        ]

    memory_count = float(len(memory_frame_indices))
    rarity = np.log((memory_count + 1.0) / np.maximum(cluster_sizes, 1.0))
    irreplaceability = nearest_distances

    scores = {}
    details = {}
    for index, frame_idx in enumerate(memory_frame_indices):
        score = float(rarity[index] * irreplaceability[index])
        if frame_idx in pinned_frames:
            score = float("inf")
        scores[frame_idx] = score
        details[frame_idx] = {
            "score": score,
            "rarity": float(rarity[index]),
            "irreplaceability": float(irreplaceability[index]),
            "cluster_id": int(cluster_ids[index]),
            "cluster_size": int(cluster_sizes[index]),
            "cluster_threshold": float(threshold),
            "cluster_rarity_neighbors": int(rarity_neighbors),
            "nearest_frame": (
                None
                if nearest_indices[index] < 0
                else int(memory_frame_indices[int(nearest_indices[index])])
            ),
            "nearest_distance": float(nearest_distances[index]),
            "irreplaceability_metric": irreplaceability_metric,
        }
    return (scores, details) if return_details else scores


def _feature_cosine_similarity(memory_frame_indices, features):
    feature_matrix = _feature_matrix(memory_frame_indices, features)
    norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
    feature_matrix = feature_matrix / np.maximum(norms, 1e-12)
    return np.clip(feature_matrix @ feature_matrix.T, -1.0, 1.0)


def _feature_cosine_similarity_cross(left_frame_indices, right_frame_indices, features):
    frame_indices = set(left_frame_indices) | set(right_frame_indices)
    missing = [idx for idx in frame_indices if idx not in features]
    if missing:
        raise ValueError(f"Missing memory features for frames: {missing[:10]}")

    left = np.stack([features[idx] for idx in left_frame_indices])
    right = np.stack([features[idx] for idx in right_frame_indices])
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    return np.clip(left @ right.T, -1.0, 1.0)


def compute_slam_covisibility_scores(
    memory_frame_indices,
    c2ws,
    pinned_frames=None,
    latent_features=None,
    n_other_observers=3,
    covisibility_threshold=0.65,
    visual_weight=0.35,
    geometry_weight=0.65,
    return_details=False,
):
    memory_frame_indices = list(memory_frame_indices)
    pinned_frames = set(pinned_frames or [])
    if not memory_frame_indices:
        return ({}, {}) if return_details else {}

    pose_distance = pose_distances(c2ws, memory_frame_indices, memory_frame_indices)
    geom_similarity = np.exp(-pose_distance)
    np.fill_diagonal(geom_similarity, 0.0)

    components = [(geometry_weight, geom_similarity)]
    if latent_features is not None:
        visual_similarity = _feature_cosine_similarity(memory_frame_indices, latent_features)
        visual_similarity = np.maximum(visual_similarity, 0.0)
        np.fill_diagonal(visual_similarity, 0.0)
        components.append((visual_weight, visual_similarity))

    total_weight = sum(weight for weight, _ in components)
    covisibility = sum(weight * matrix for weight, matrix in components) / max(total_weight, 1e-12)
    np.fill_diagonal(covisibility, 0.0)

    scores = {}
    details = {}
    for row, frame_idx in enumerate(memory_frame_indices):
        row_values = covisibility[row]
        observer_indices = np.flatnonzero(row_values >= covisibility_threshold)
        covisible_observers = int(observer_indices.size)
        redundancy_ratio = min(covisible_observers / max(float(n_other_observers), 1.0), 1.0)

        if row_values.size:
            nearest_index = int(np.argmax(row_values))
            nearest_frame = int(memory_frame_indices[nearest_index])
            max_covisibility = float(row_values[nearest_index])
        else:
            nearest_frame = None
            max_covisibility = 0.0

        marginal_contribution = 1.0 / (covisible_observers + 1.0)
        unique_bonus = 1.0 - max_covisibility
        score = (1.0 - redundancy_ratio) + 0.5 * marginal_contribution + 0.25 * unique_bonus
        if frame_idx in pinned_frames:
            score = float("inf")

        scores[frame_idx] = float(score)
        details[frame_idx] = {
            "score": float(score),
            "redundancy_ratio": float(redundancy_ratio),
            "covisible_observers": covisible_observers,
            "max_covisibility": float(max_covisibility),
            "nearest_covisible_frame": nearest_frame,
            "marginal_contribution": float(marginal_contribution),
            "unique_bonus": float(unique_bonus),
            "covisibility_threshold": float(covisibility_threshold),
            "n_other_observers": int(n_other_observers),
        }

    return (scores, details) if return_details else scores


def _min_max_normalize_scores(frame_indices, raw_scores):
    values = np.asarray([raw_scores[idx] for idx in frame_indices], dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {idx: 0.0 for idx in frame_indices}

    low = float(finite.min())
    high = float(finite.max())
    span = high - low
    normalized = {}
    for frame_idx, value in zip(frame_indices, values):
        if not np.isfinite(value):
            normalized[frame_idx] = 1.0
        elif span <= 1e-12:
            normalized[frame_idx] = 1.0
        else:
            normalized[frame_idx] = (float(value) - low) / span
    return normalized


def compute_coverage_ri_fusion_scores(
    memory_frame_indices,
    c2ws,
    latent_features=None,
    rarity_features=None,
    irreplaceability_features=None,
    irreplaceability_metric="cosine",
    pinned_frames=None,
    coverage_weight=0.75,
    rarity_neighbors=3,
    return_details=False,
):
    """Fuse independently normalized geometric-coverage and RI utilities.

    The admission decision is intentionally not part of this function. It scores
    only candidates admitted by the active policy before retention.
    """
    memory_frame_indices = list(memory_frame_indices)
    pinned_frames = set(pinned_frames or [])
    coverage_weight = float(coverage_weight)
    if not 0.0 <= coverage_weight <= 1.0:
        raise ValueError("coverage_weight must be in [0, 1]")
    if not memory_frame_indices:
        return ({}, {}) if return_details else {}

    coverage_scores, coverage_details = compute_slam_covisibility_scores(
        memory_frame_indices=memory_frame_indices,
        c2ws=c2ws,
        pinned_frames=None,
        latent_features=latent_features,
        return_details=True,
    )
    ri_scores, ri_details = compute_rarity_irreplaceability_scores(
        memory_frame_indices=memory_frame_indices,
        latent_features=latent_features,
        rarity_features=rarity_features,
        irreplaceability_features=irreplaceability_features,
        irreplaceability_metric=irreplaceability_metric,
        pinned_frames=None,
        rarity_neighbors=rarity_neighbors,
        return_details=True,
    )
    coverage_normalized = _min_max_normalize_scores(
        memory_frame_indices, coverage_scores
    )
    ri_normalized = _min_max_normalize_scores(memory_frame_indices, ri_scores)

    scores = {}
    details = {}
    for frame_idx in memory_frame_indices:
        pinned = frame_idx in pinned_frames
        fused = (
            coverage_weight * coverage_normalized[frame_idx]
            + (1.0 - coverage_weight) * ri_normalized[frame_idx]
        )
        score = float("inf") if pinned else float(fused)
        scores[frame_idx] = score
        details[frame_idx] = {
            "score": score,
            "fusion_pinned": pinned,
            "fusion_coverage_weight": coverage_weight,
            "fusion_ri_weight": 1.0 - coverage_weight,
            "fusion_coverage_raw": float(coverage_scores[frame_idx]),
            "fusion_coverage_normalized": float(
                coverage_normalized[frame_idx]
            ),
            "fusion_ri_raw": float(ri_scores[frame_idx]),
            "fusion_ri_normalized": float(ri_normalized[frame_idx]),
            "rarity": ri_details[frame_idx]["rarity"],
            "irreplaceability": ri_details[frame_idx]["irreplaceability"],
            "cluster_id": ri_details[frame_idx]["cluster_id"],
            "cluster_size": ri_details[frame_idx]["cluster_size"],
            "cluster_threshold": ri_details[frame_idx]["cluster_threshold"],
            "cluster_rarity_neighbors": ri_details[frame_idx][
                "cluster_rarity_neighbors"
            ],
            "nearest_frame": ri_details[frame_idx]["nearest_frame"],
            "nearest_distance": ri_details[frame_idx]["nearest_distance"],
            "redundancy_ratio": coverage_details[frame_idx]["redundancy_ratio"],
            "covisible_observers": coverage_details[frame_idx][
                "covisible_observers"
            ],
            "max_covisibility": coverage_details[frame_idx]["max_covisibility"],
            "nearest_covisible_frame": coverage_details[frame_idx][
                "nearest_covisible_frame"
            ],
            "marginal_contribution": coverage_details[frame_idx][
                "marginal_contribution"
            ],
            "unique_bonus": coverage_details[frame_idx]["unique_bonus"],
        }
    return (scores, details) if return_details else scores


def compute_kcenter_coreset_scores(
    memory_frame_indices,
    archive_frame_indices,
    c2ws,
    budget,
    pinned_frames=None,
    latent_features=None,
    visual_weight=0.5,
    pose_weight=0.5,
    time_weight=0.0,
    return_details=False,
):
    """Select retained memory frames by greedy k-center coverage.

    The archive is the historical trajectory to cover. The candidate memory set
    contains frames the model can keep and retrieve. Higher returned scores mean
    "keep"; unselected frames receive low scores and are evicted by the buffer.
    """
    memory_frame_indices = [int(idx) for idx in memory_frame_indices]
    archive_frame_indices = [int(idx) for idx in archive_frame_indices]
    pinned_frames = {int(idx) for idx in (pinned_frames or [])}
    if budget is None:
        raise ValueError("kcenter_coreset requires an explicit memory budget")
    if budget <= 0:
        raise ValueError("kcenter_coreset budget must be positive")
    if not memory_frame_indices:
        return ({}, {}) if return_details else {}
    if not archive_frame_indices:
        archive_frame_indices = list(memory_frame_indices)

    use_visual = latent_features is not None and float(visual_weight) > 0.0
    if len(memory_frame_indices) <= budget:
        scores = {
            frame_idx: float("inf") if frame_idx in pinned_frames else 1.0
            for frame_idx in memory_frame_indices
        }
        details = {
            frame_idx: {
                "score": scores[frame_idx],
                "kcenter_selected": True,
                "kcenter_forced_keep": frame_idx in pinned_frames,
                "kcenter_rank": index,
                "kcenter_radius": 0.0,
                "kcenter_mean_radius": 0.0,
                "kcenter_removal_radius_increase": None,
                "kcenter_archive_size": len(archive_frame_indices),
                "kcenter_nearest_archive_frame": None,
                "kcenter_nearest_archive_distance": None,
                "kcenter_selected_for_archive_frame": None,
                "kcenter_visual_weight": float(visual_weight if use_visual else 0.0),
                "kcenter_pose_weight": float(pose_weight),
                "kcenter_time_weight": float(time_weight),
            }
            for index, frame_idx in enumerate(memory_frame_indices)
        }
        return (scores, details) if return_details else scores

    components = []
    if use_visual:
        visual_similarity = _feature_cosine_similarity_cross(
            archive_frame_indices,
            memory_frame_indices,
            latent_features,
        )
        visual_distance = (1.0 - visual_similarity) / 2.0
        visual_distance = np.clip(visual_distance, 0.0, 1.0)
        components.append((float(visual_weight), visual_distance))

    if pose_weight:
        pose_distance = pose_distances(c2ws, archive_frame_indices, memory_frame_indices)
        pose_distance = 1.0 - np.exp(-pose_distance)
        components.append((float(pose_weight), pose_distance))

    if time_weight:
        archive_times = np.asarray(archive_frame_indices, dtype=np.float64)
        memory_times = np.asarray(memory_frame_indices, dtype=np.float64)
        time_scale = max(
            float(max(max(archive_frame_indices), max(memory_frame_indices)) + 1),
            1.0,
        )
        time_distance = np.abs(archive_times[:, None] - memory_times[None, :]) / time_scale
        components.append((float(time_weight), time_distance))

    if not components:
        raise ValueError("kcenter_coreset needs at least one positive distance component")

    total_weight = max(sum(weight for weight, _ in components), 1e-12)
    distance = sum(weight * matrix for weight, matrix in components) / total_weight

    frame_to_col = {frame_idx: col for col, frame_idx in enumerate(memory_frame_indices)}
    forced_cols = [
        frame_to_col[frame_idx]
        for frame_idx in memory_frame_indices
        if frame_idx in pinned_frames
    ]

    selected_cols = []
    selected_set = set()
    selected_by_archive = {}

    for col in forced_cols:
        if col not in selected_set:
            selected_set.add(col)
            selected_cols.append(col)
            selected_by_archive[col] = None

    if selected_cols:
        covered_distance = np.min(distance[:, selected_cols], axis=1)
    else:
        first_col = int(np.argmin(np.mean(distance, axis=0)))
        selected_set.add(first_col)
        selected_cols.append(first_col)
        selected_by_archive[first_col] = None
        covered_distance = distance[:, first_col].copy()

    while len(selected_cols) < min(int(budget), len(memory_frame_indices)):
        farthest_archive_row = int(np.argmax(covered_distance))
        candidate_order = np.argsort(distance[farthest_archive_row])
        best_col = None
        for col in candidate_order:
            col = int(col)
            if col not in selected_set:
                best_col = col
                break
        if best_col is None:
            break

        selected_set.add(best_col)
        selected_cols.append(best_col)
        selected_by_archive[best_col] = int(archive_frame_indices[farthest_archive_row])
        covered_distance = np.minimum(covered_distance, distance[:, best_col])

    selected_frames = [memory_frame_indices[col] for col in selected_cols]
    selected_frame_set = set(selected_frames)
    current_radius = float(np.max(covered_distance)) if covered_distance.size else 0.0
    mean_radius = float(np.mean(covered_distance)) if covered_distance.size else 0.0

    removal_radius_increases = {}
    for col in selected_cols:
        other_cols = [other for other in selected_cols if other != col]
        if other_cols:
            without_col = np.min(distance[:, other_cols], axis=1)
            without_radius = float(np.max(without_col))
        else:
            without_radius = float("inf")
        removal_radius_increases[col] = without_radius - current_radius

    scores = {}
    details = {}
    for col, frame_idx in enumerate(memory_frame_indices):
        selected = frame_idx in selected_frame_set
        forced = frame_idx in pinned_frames
        if forced:
            score = float("inf")
        elif selected:
            score = 1.0 + max(float(removal_radius_increases.get(col, 0.0)), 0.0)
        else:
            score = -1.0

        nearest_archive_row = int(np.argmin(distance[:, col])) if distance.shape[0] else None
        selected_archive_frame = selected_by_archive.get(col)
        rank = selected_frames.index(frame_idx) if selected else None
        scores[frame_idx] = float(score)
        details[frame_idx] = {
            "score": float(score),
            "kcenter_selected": bool(selected),
            "kcenter_forced_keep": bool(forced),
            "kcenter_rank": rank,
            "kcenter_radius": current_radius,
            "kcenter_mean_radius": mean_radius,
            "kcenter_removal_radius_increase": (
                float(removal_radius_increases.get(col, 0.0)) if selected else 0.0
            ),
            "kcenter_archive_size": len(archive_frame_indices),
            "kcenter_nearest_archive_frame": (
                None if nearest_archive_row is None else int(archive_frame_indices[nearest_archive_row])
            ),
            "kcenter_nearest_archive_distance": (
                None if nearest_archive_row is None else float(distance[nearest_archive_row, col])
            ),
            "kcenter_selected_for_archive_frame": selected_archive_frame,
            "kcenter_visual_weight": float(visual_weight if use_visual else 0.0),
            "kcenter_pose_weight": float(pose_weight),
            "kcenter_time_weight": float(time_weight),
        }

    return (scores, details) if return_details else scores


def _historical_query_medoids(memory_frame_indices, latent_features, rarity_neighbors=3):
    """Cluster candidates by content-feature similarity and return one medoid per cluster.

    Reuses the same clustering primitives as ``compute_rarity_irreplaceability_scores``
    so "distinct scene content" means the same thing across policies. Each cluster
    contributes exactly one query point regardless of its size -- a region visited
    many times must not outweigh a rare region in the coverage objective (that is
    what lets MCE beat a "concentrate on frequently-reused anchors" heuristic).

    ``rarity_neighbors`` controls how coarse that clustering is (see
    ``estimate_cluster_threshold``): k=1 is the tightest possible threshold and
    was, until this parameter existed, the only granularity MCE's Q_hist could
    ever use here. Real access-trace data from a WorldMem budget sweep showed
    the number of distinct medoids sitting well below budget at every level
    (e.g. a median of ~11 medoids at budget=16 up to ~78 at budget=128) --
    every candidate beyond its cluster's medoid gets a near-zero, barely
    distinguishable marginal (see the coverage objective's docstring), so a
    persistently low k=1 threshold means most of the retained budget is
    filled by an effectively arbitrary tiebreak among redundant candidates.
    Larger k coarsens clustering and shrinks that gap, at the cost of treating
    more distinct content as "the same query" -- the right value is a real
    open question, not assumed here.
    """
    memory_frame_indices = list(memory_frame_indices)
    if len(memory_frame_indices) == 1:
        return memory_frame_indices

    feature_matrix = _feature_matrix(memory_frame_indices, latent_features)
    pairwise = cosine_distances(feature_matrix)
    np.fill_diagonal(pairwise, np.inf)
    threshold = estimate_cluster_threshold(pairwise, rarity_neighbors)

    cluster_pairwise = pairwise.copy()
    np.fill_diagonal(cluster_pairwise, 0.0)
    _, clusters = connected_components_from_threshold(cluster_pairwise, threshold=threshold)

    medoid_positions = []
    for members in clusters:
        if len(members) == 1:
            medoid_positions.append(members[0])
            continue
        sub_distances = cluster_pairwise[np.ix_(members, members)]
        total_distance = sub_distances.sum(axis=1)
        medoid_positions.append(members[int(np.argmin(total_distance))])

    return [memory_frame_indices[position] for position in medoid_positions]


def compute_marginal_coverage_eviction_scores(
    memory_frame_indices,
    c2ws,
    budget,
    pinned_frames=None,
    latent_features=None,
    alpha=0.65,
    rarity_neighbors=3,
    return_details=False,
):
    """Marginal Coverage Eviction (MCE): write-path noisy-OR set-coverage eviction.

    This is a bounded-memory *eviction* policy, not a retrieval rule: given the
    full prospective candidate pool (current memory plus newly admitted frames)
    it decides which frames to keep so the retained set best covers the scene
    content already generated.

    Objective. Each query q (see below) has weight w_q (sum to 1) and each
    candidate m has a kernel K(q, m) in [0, 1]. Retaining set M gives coverage

        U(M) = sum_q w_q * (1 - prod_{m in M} (1 - K(q, m)))

    -- a standard noisy-OR / probabilistic-set-cover construction (monotone
    submodular), not new math here. The exact marginal loss of removing item i
    from pool P is

        Delta_i(P) = sum_q w_q * K(q, i) * prod_{x in P \\ {i}} (1 - K(q, x))

    Query set: historical-only (Q_hist), no anticipated-future term. An earlier
    version of this method mixed in a Q_ctrl term for a known future camera
    path; it is deliberately dropped here. Two reasons: (1) empirically the
    historical-only configuration matched or beat future-weighted variants on
    LPIPS / pose-reconstruction ATE at bounded budget; (2) this is an online
    setting -- the future state sequence is only revealed incrementally, so
    committing coverage weight to one anticipated continuation is a point bet
    that is wasted if wrong, whereas covering diverse *past* content hedges
    against whatever comes next. Q_hist is built by clustering the candidates
    by content-feature similarity (one connected-components cluster per
    distinct scene mode) and taking one medoid per cluster as a query, weighted
    uniformly (not by cluster size) -- see ``_historical_query_medoids``.

    Kernel: an explicit convex combination, not a product,

        K(q, m) = alpha * K_geo(q, m) + (1 - alpha) * K_vis(q, m)

    so that strong agreement on one cue (e.g. K_geo = 1 for a near-identical
    camera pose) is not zeroed out by a weak second cue. K_geo reuses this
    file's existing pose-distance-to-similarity convention (``exp(-pose
    distance)``, the same transform ``compute_slam_covisibility_scores`` uses)
    since WorldMem's memory is camera-pose indexed via ``c2ws``. K_vis is
    cosine similarity between ``latent_features`` (whichever content embedding
    ``memory_feature_backend`` produced -- DINO features give the most
    semantically meaningful K_vis; raw VAE latents work but are a weaker cue),
    calibrated from [-1, 1] to [0, 1]. Pass ``alpha=0`` to drop K_geo entirely
    (e.g. for a state representation with no meaningful pose notion).

    Algorithm: reverse deletion (Algorithm 1), not forward-greedy addition.
    Starting from the full candidate pool, repeatedly evict
    argmin_i Delta_i(P) until |P| <= budget, recomputing exact marginals after
    each removal. This is the natural fit for a write-path policy that already
    holds a full candidate set and must shrink it -- it does *not* carry the
    classic offline forward-greedy (1 - 1/e) approximation guarantee, which is
    a different algorithm; treat this as a well-motivated heuristic, not a
    proven bound, unless a streaming/deletion-robust submodular bound is
    established for this setting.

    Efficiency: the running product P_q = prod_{m in P} (1 - K(q, m)) is
    cached once per pool and updated on each eviction rather than recomputed
    from scratch, giving O(budget * |Q|) total work instead of O(budget^2 *
    |Q|). It is tracked as a sum of logs, not a direct-space product: removing
    an item divides P_q by (1 - K(q, removed)), and with hundreds of
    candidates trimmed down to a small budget (WorldMem's initial-context
    pool can be ~600 frames wide), that is hundreds of divisions by values as
    small as the 1e-6 kernel-clip floor. Direct-space accumulation only stays
    below "dozens" of candidates before repeated division-by-near-zero
    compounds P_q past float64's max and overflows to inf; log-space turns
    that same update into subtraction of bounded values, which cannot.

    Forced-keep frames (``pinned_frames``, e.g. frame 0) remain in the pool
    and keep contributing coverage, but are never eviction candidates.
    """
    memory_frame_indices = [int(idx) for idx in memory_frame_indices]
    pinned_frames = {int(idx) for idx in (pinned_frames or [])}
    if budget is None:
        raise ValueError("mce requires an explicit memory budget")
    if budget <= 0:
        raise ValueError("mce budget must be positive")
    if not memory_frame_indices:
        return ({}, {}) if return_details else {}
    if len(set(memory_frame_indices)) != len(memory_frame_indices):
        raise ValueError("mce candidates must be unique")
    if latent_features is None:
        raise ValueError("mce requires content latent_features")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("mce alpha must be in [0, 1]")

    candidate_set = set(memory_frame_indices)
    unknown_pinned = pinned_frames - candidate_set
    if unknown_pinned:
        raise ValueError(f"mce pinned frames are not candidates: {sorted(unknown_pinned)[:10]}")
    if len(pinned_frames) > budget:
        raise ValueError("mce has more pinned frames than its budget")

    _feature_matrix(memory_frame_indices, latent_features)  # validates presence

    num_candidates = len(memory_frame_indices)
    selected_limit = min(int(budget), num_candidates)

    # --- Query set: historical medoids only (Q = Q_hist, no future term) ---
    query_frame_indices = _historical_query_medoids(
        memory_frame_indices, latent_features, rarity_neighbors
    )
    num_queries = len(query_frame_indices)

    # --- Kernel: explicit convex combination, not a product -----------------
    if alpha > 0.0:
        geo_distance = pose_distances(c2ws, query_frame_indices, memory_frame_indices)
        geo_kernel = np.exp(-geo_distance)
    else:
        geo_kernel = np.zeros((num_queries, num_candidates), dtype=np.float64)
    vis_cosine = _feature_cosine_similarity_cross(
        query_frame_indices, memory_frame_indices, latent_features
    )
    vis_kernel = np.clip((vis_cosine + 1.0) / 2.0, 0.0, 1.0)
    kernel = np.clip(alpha * geo_kernel + (1.0 - alpha) * vis_kernel, 0.0, 1.0 - 1e-6)
    weights = np.full(num_queries, 1.0 / max(num_queries, 1), dtype=np.float64)

    # --- Algorithm 1: reverse deletion with a log-space running product ----
    # log(1 - K) is finite everywhere: kernel is clipped to <= 1 - 1e-6, so
    # one_minus_kernel is bounded away from 0 and this never hits log(0).
    frame_to_col = {frame_idx: col for col, frame_idx in enumerate(memory_frame_indices)}
    forced_cols = {frame_to_col[frame_idx] for frame_idx in pinned_frames}
    remaining_cols = list(range(num_candidates))
    one_minus_kernel = 1.0 - kernel
    log_one_minus_kernel = np.log(one_minus_kernel)
    log_pool_product = np.sum(log_one_minus_kernel, axis=1)  # log P_q over the full initial pool

    removal_order = []
    removal_marginals = {}
    while len(remaining_cols) > selected_limit:
        remaining = np.array(remaining_cols)
        # exp(log_pool_product - log(1 - K(q, i))) == P_q over pool \ {i}, computed
        # by subtraction (bounded) instead of dividing by a near-zero denominator.
        pool_product_excluding = np.exp(
            log_pool_product[:, None] - log_one_minus_kernel[:, remaining]
        )
        marginals = np.sum(
            weights[:, None] * kernel[:, remaining] * pool_product_excluding,
            axis=0,
        )
        assert np.all(np.isfinite(marginals)), "mce marginals contain NaN/Inf"
        eviction_candidates = [
            (float(marginals[position]), col)
            for position, col in enumerate(remaining_cols)
            if col not in forced_cols
        ]
        if not eviction_candidates:
            break
        loss, evict_col = min(eviction_candidates, key=lambda item: item[0])
        removal_order.append(evict_col)
        removal_marginals[evict_col] = loss
        log_pool_product = log_pool_product - log_one_minus_kernel[:, evict_col]
        remaining_cols.remove(evict_col)

    selected_cols = remaining_cols
    selected_frame_set = {memory_frame_indices[col] for col in selected_cols}

    # Final leave-one-out marginal for each survivor, w.r.t. the final set --
    # this is the reported "value", not what drove the eviction decisions.
    final_uncovered = (
        np.prod(one_minus_kernel[:, selected_cols], axis=1)
        if selected_cols
        else np.ones(kernel.shape[0])
    )
    survivor_marginals = {}
    for col in selected_cols:
        denom = np.maximum(one_minus_kernel[:, col], 1e-12)
        survivor_marginals[col] = float(np.sum(weights * kernel[:, col] * final_uncovered / denom))
    coverage_value = float(np.sum(weights * (1.0 - final_uncovered)))

    scores = {}
    details = {}
    for col, frame_idx in enumerate(memory_frame_indices):
        selected = frame_idx in selected_frame_set
        forced = frame_idx in pinned_frames
        if forced:
            score = float("inf")
        elif selected:
            score = 1.0 + survivor_marginals.get(col, 0.0)
        else:
            score = -1.0 - removal_marginals.get(col, 0.0)

        scores[frame_idx] = float(score)
        details[frame_idx] = {
            "score": float(score),
            "mce_selected": bool(selected),
            "mce_forced_keep": bool(forced),
            "mce_removal_rank": (
                removal_order.index(col) if col in removal_order else None
            ),
            "mce_removal_marginal": removal_marginals.get(col),
            "mce_survivor_marginal": survivor_marginals.get(col),
            "mce_coverage_value": coverage_value,
            "mce_alpha": float(alpha),
            "mce_rarity_neighbors": int(rarity_neighbors),
            "mce_num_queries": num_queries,
            "mce_query_frames": [int(f) for f in query_frame_indices],
        }

    return (scores, details) if return_details else scores
