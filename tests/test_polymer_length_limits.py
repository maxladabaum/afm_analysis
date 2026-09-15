import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import origami_counter_app as app


class PolymerLengthLimitTests(unittest.TestCase):
    def test_long_contour_excluded_from_measurements(self):
        rgb = np.zeros((300, 300, 3), dtype=np.uint8)
        rgb[70:74, 60:130] = 255
        rgb[120:124, 60:260] = 255
        with patch.object(app, 'scan_bbox', return_value=(0, 0, 300, 300)):
            unlimited = app.analyze_polymer_image(rgb, 900, 20, 5, 0, lambda *_: None)
            limited = app.analyze_polymer_image(rgb, 900, 20, 5, 0, lambda *_: None, max_length_nm=100)
        self.assertEqual(sum(not obj.excluded_reason for obj in unlimited.objects), 2)
        accepted = [obj for obj in limited.objects if not obj.excluded_reason]
        rejected = [obj for obj in limited.objects if obj.excluded_reason == 'longer_than_max_length']
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 1)
        self.assertLess(accepted[0].length_nm, 100)
        self.assertGreater(rejected[0].length_nm, 100)
        self.assertTrue(all(row['polymer_ids'] == str(accepted[0].object_id) for row in limited.msd_rows))
        self.assertNotEqual(limited.params, unlimited.params)

    def params(self, maximum):
        def field(value):
            return SimpleNamespace(get=lambda: value)
        fake = SimpleNamespace(polymer_min_length_nm=field('80'), polymer_segment_nm=field('20'),
                               threshold_bias=field('0'), polymer_max_length_nm=field(maximum), polymer_branch_prune_nm=field('0'))
        return app.OrigamiCounterApp.current_polymer_analysis_params(fake)

    def test_blank_is_unlimited_and_equal_limits_allowed(self):
        self.assertEqual(self.params(' ')[3], math.inf)
        self.assertEqual(self.params('80')[3], 80)
        self.assertEqual(self.params('500')[3], 500)

    def test_invalid_limits_rejected(self):
        for value in ('79', '0', '-1', 'nan', 'inf', 'text'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.params(value)


if __name__ == '__main__':
    unittest.main()
