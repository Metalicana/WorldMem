import unittest

from utils.evaluate_worldmem_rfid import (
    build_frame_index_plan,
    distribute_frame_counts,
    evenly_spaced_indices,
)


class WorldMemRFIDTest(unittest.TestCase):
    def test_distributes_5000_frames_over_15_videos(self):
        counts = distribute_frame_counts(5000, 15, 600)
        self.assertEqual(sum(counts), 5000)
        self.assertEqual(counts[:5], [334] * 5)
        self.assertEqual(counts[5:], [333] * 10)

    def test_even_indices_are_unique_and_cover_horizon(self):
        indices = evenly_spaced_indices(600, 334)
        self.assertEqual(len(indices), 334)
        self.assertEqual(len(set(indices)), 334)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 599)

    def test_index_plan_uses_exact_gt_offset(self):
        plan = build_frame_index_plan(
            batch_ids=range(15),
            duration_frames=600,
            total_frames=5000,
            gt_frame_offset=700,
        )
        self.assertEqual(len(plan), 15)
        self.assertEqual(sum(row["sampled_frames"] for row in plan), 5000)
        for row in plan:
            self.assertEqual(
                row["gt_frame_indices"],
                [700 + index for index in row["generated_frame_indices"]],
            )

    def test_rejects_more_frames_than_available(self):
        with self.assertRaises(ValueError):
            distribute_frame_counts(9001, 15, 600)


if __name__ == "__main__":
    unittest.main()
