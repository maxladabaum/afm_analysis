import unittest

import numpy as np

import origami_counter_app as app


class BranchPruningTests(unittest.TestCase):
    def test_adjustable_cutoff_removes_short_spur_and_preserves_backbone(self):
        mask = np.zeros((100, 120), dtype=bool)
        mask[50, 10:110] = True
        mask[45:51, 60] = True
        strict, count = app.prune_short_skeleton_branches(mask, 0)
        np.testing.assert_array_equal(strict, mask)
        self.assertEqual(count, 0)
        too_small, count = app.prune_short_skeleton_branches(mask, 3)
        self.assertIsNone(app.ordered_skeleton_path(too_small))
        self.assertEqual(count, 0)
        cleaned, count = app.prune_short_skeleton_branches(mask, 6)
        path = app.ordered_skeleton_path(cleaned)
        self.assertIsNotNone(path)
        self.assertEqual(count, 1)
        self.assertEqual({path[0], path[-1]}, {(50, 10), (50, 109)})
        self.assertTrue(mask[45, 60])  # Input is not mutated.

    def test_substantial_branch_is_not_removed(self):
        mask = np.zeros((100, 120), dtype=bool)
        mask[50, 10:110] = True
        mask[20:51, 60] = True
        cleaned, count = app.prune_short_skeleton_branches(mask, 10)
        np.testing.assert_array_equal(cleaned, mask)
        self.assertEqual(count, 0)
        self.assertIsNone(app.ordered_skeleton_path(cleaned))

    def test_unbranched_short_polymer_is_not_pruned(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[15, 10:20] = True
        cleaned, count = app.prune_short_skeleton_branches(mask, 100)
        np.testing.assert_array_equal(cleaned, mask)
        self.assertEqual(count, 0)

    def test_loop_is_not_converted_to_open_path(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[5, 5:25] = mask[24, 5:25] = True
        mask[5:25, 5] = mask[5:25, 24] = True
        cleaned, count = app.prune_short_skeleton_branches(mask, 100)
        self.assertEqual(count, 0)
        self.assertIsNone(app.ordered_skeleton_path(cleaned))


if __name__ == '__main__':
    unittest.main()
