import math
from collections import OrderedDict

import numpy as np


SUPPORTED_MEMORY_POLICIES = (
    "unbounded",
    "fifo",
    "rarity_irreplaceability",
    "slam_covisibility",
    "kcenter_coreset",
)
BUDGETED_MEMORY_POLICIES = (
    "fifo",
    "rarity_irreplaceability",
    "slam_covisibility",
    "kcenter_coreset",
)


class FrameMemoryBuffer:
    def __init__(self, policy="unbounded", budget=None, pinned_frames=None):
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
            else:
                evicted_frame_idx = min(
                    evictable,
                    key=lambda idx: (
                        self._stats[idx].get("score", 0.0),
                        self._stats[idx]["insert_order"],
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


def cosine_distances(features):
    features = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-12)
    similarities = np.clip(features @ features.T, -1.0, 1.0)
    return 1.0 - similarities


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


def estimate_cluster_threshold(pairwise_distances):
    finite = pairwise_distances[np.isfinite(pairwise_distances)]
    if finite.size == 0:
        return 0.0
    nearest = np.partition(pairwise_distances, 0, axis=1)[:, 0]
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
    latent_features,
    pinned_frames=None,
    return_details=False,
):
    memory_frame_indices = list(memory_frame_indices)
    pinned_frames = set(pinned_frames or [])
    if not memory_frame_indices:
        return ({}, {}) if return_details else {}

    feature_matrix = _feature_matrix(memory_frame_indices, latent_features)
    pairwise = cosine_distances(feature_matrix)
    np.fill_diagonal(pairwise, np.inf)

    if len(memory_frame_indices) == 1:
        cluster_ids = np.zeros(1, dtype=np.int64)
        cluster_sizes = np.ones(1, dtype=np.float64)
        threshold = 0.0
        nearest_distances = np.ones(1, dtype=np.float64)
        nearest_indices = np.full(1, -1, dtype=np.int64)
    else:
        threshold = estimate_cluster_threshold(pairwise)
        cluster_pairwise = pairwise.copy()
        np.fill_diagonal(cluster_pairwise, 0.0)
        cluster_ids, clusters = connected_components_from_threshold(
            cluster_pairwise,
            threshold=threshold,
        )
        cluster_sizes = np.array([len(clusters[cluster_id]) for cluster_id in cluster_ids])
        nearest_indices = np.argmin(pairwise, axis=1)
        nearest_distances = pairwise[np.arange(len(memory_frame_indices)), nearest_indices]

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
            "nearest_frame": (
                None
                if nearest_indices[index] < 0
                else int(memory_frame_indices[int(nearest_indices[index])])
            ),
            "nearest_distance": float(nearest_distances[index]),
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
