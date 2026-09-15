import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import origami_counter_app as app


class PolymerBoundaryTests(unittest.TestCase):
    def test_scan_frame_does_not_extend_fit_range_at_different_sizes(self):
        # An open L-shaped frame is a valid unbranched skeleton, but not a polymer.
        original = np.zeros((300, 300, 3), dtype=np.uint8)
        original[:245, :4] = 255
        original[241:245, :245] = 255
        original[70:74, 60:130] = 255
        original[120:124, 100:170] = 255
        for factor in (0.5, 1, 2):
            with self.subTest(factor=factor):
                side = int(300 * factor)
                rgb = np.asarray(Image.fromarray(original).resize((side, side), Image.Resampling.NEAREST))
                with patch.object(app, 'scan_bbox', return_value=(0, 0, side, side)):
                    objects = app.detect_polymer_contours(rgb, 900 * factor, 20)
                accepted = [obj for obj in objects if not obj.excluded_reason]
                boundary = [obj for obj in objects if obj.excluded_reason == 'touches_scan_boundary']
                self.assertEqual(len(accepted), 2)
                self.assertEqual(len(boundary), 1)
                self.assertGreater(boundary[0].length_nm, 500)
                self.assertTrue(all(65 < obj.length_nm < 85 for obj in accepted))
                rows = app.polymer_msd_table(objects, 900 * factor, 5)
                self.assertLessEqual(max(row['contour_separation_nm'] for row in rows), 85)
                self.assertTrue(all(str(boundary[0].object_id) not in row['polymer_ids'].split(';')
                                    for row in rows))

    def test_polymer_cut_off_at_edge_is_excluded(self):
        rgb = np.zeros((200, 200, 3), dtype=np.uint8)
        rgb[90:95, :100] = 255
        with patch.object(app, 'scan_bbox', return_value=(0, 0, 200, 200)):
            objects = app.detect_polymer_contours(rgb, 900, 20)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].excluded_reason, 'touches_scan_boundary')


if __name__ == '__main__':
    unittest.main()
