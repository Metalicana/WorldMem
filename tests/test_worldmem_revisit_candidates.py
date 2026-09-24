import importlib.util
import tempfile
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "export_worldmem_revisit_candidates.py"
SPEC = importlib.util.spec_from_file_location("export_worldmem_revisit_candidates", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def c2w(x=0.0, yaw_deg=0.0):
    yaw = np.deg2rad(yaw_deg)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ]
    )
    matrix[0, 3] = x
    return matrix


def test_rotation_angle_handles_wraparound():
    assert MODULE.rotation_angle_deg(c2w(yaw_deg=179)[:3, :3], c2w(yaw_deg=-179)[:3, :3]) < 3


def test_return_requires_and_uses_real_departure():
    poses = []
    poses.extend(c2w(x=0.0) for _ in range(6))
    poses.extend(c2w(x=3.0, yaw_deg=60) for _ in range(12))
    poses.extend(c2w(x=0.0) for _ in range(8))
    rows = MODULE.find_verified_returns(
        np.stack(poses),
        trajectory_fps=2,
        min_separation_sec=5,
        endpoint_position_threshold=0.75,
        endpoint_rotation_threshold_deg=15,
        departure_position_threshold=2.0,
        departure_rotation_threshold_deg=45,
        min_away_sec=1,
    )
    assert rows
    row = max(rows, key=lambda item: item["temporal_separation_sec"])
    assert 6 <= row["middle_frame"] < 18
    assert row["away_duration_sec"] >= 6
    assert row["endpoint_position_distance"] == 0
    assert row["endpoint_rotation_deg"] == 0


def test_similar_pose_without_departure_is_not_a_revisit():
    poses = np.stack([c2w(x=0.1 * np.sin(index / 4)) for index in range(40)])
    rows = MODULE.find_verified_returns(
        poses,
        trajectory_fps=2,
        min_separation_sec=5,
        endpoint_position_threshold=0.75,
        endpoint_rotation_threshold_deg=15,
        departure_position_threshold=2.0,
        departure_rotation_threshold_deg=45,
        min_away_sec=1,
    )
    assert rows == []


def test_return_episode_does_not_turn_loiter_frames_into_new_revisits():
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 600, axis=0)
    poses[100:200, 0, 3] = 3.0
    rows = MODULE.find_verified_returns(
        poses,
        trajectory_fps=10,
        min_separation_sec=5,
        endpoint_position_threshold=0.75,
        endpoint_rotation_threshold_deg=15,
        departure_position_threshold=2.0,
        departure_rotation_threshold_deg=45,
        min_away_sec=1,
    )
    row = next(item for item in rows if item["first_frame"] == 0)
    assert row["revisit_frame"] == 200
    assert row["revisit_episode_start_frame"] == 200
    assert row["revisit_episode_end_frame_exclusive"] == 600


def test_ranking_suppresses_duplicate_return_times():
    rows = [
        {
            "first_frame": 0,
            "middle_frame": 20,
            "revisit_frame": 50,
            "endpoint_error_normalized": 0.2,
            "temporal_separation_sec": 5.0,
            "away_duration_sec": 2.0,
        },
        {
            "first_frame": 1,
            "middle_frame": 21,
            "revisit_frame": 53,
            "endpoint_error_normalized": 0.3,
            "temporal_separation_sec": 5.2,
            "away_duration_sec": 2.0,
        },
        {
            "first_frame": 2,
            "middle_frame": 40,
            "revisit_frame": 80,
            "endpoint_error_normalized": 0.4,
            "temporal_separation_sec": 7.8,
            "away_duration_sec": 3.0,
        },
    ]
    selected = MODULE.select_ranked_candidates(
        rows,
        max_candidates=10,
        revisit_suppression_frames=10,
    )
    assert [row["revisit_frame"] for row in selected] == [50, 80]


def test_generated_pose_mapping_excludes_context_and_initial_skip():
    with tempfile.TemporaryDirectory() as directory:
        video = Path(directory) / "trajectory.mp4"
        poses = np.zeros((705, 5), dtype=np.float64)
        poses[:, 0] = np.arange(len(poses))
        actions = np.zeros((705, 8), dtype=np.int64)
        np.savez(video.with_suffix(".npz"), poses=poses, actions=actions)

        c2ws = MODULE.load_exact_generated_c2ws(
            video,
            output_frames=5,
            context_frames=600,
            source_start_offset=100,
        )
        assert np.allclose(c2ws[:, 0, 3], np.arange(600, 605))
