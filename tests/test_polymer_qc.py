import math
import unittest
import tkinter as tk
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

import origami_counter_app as app


class PolymerQCTests(unittest.TestCase):
    def test_qc_text_is_drawn_above_preview_and_can_be_hidden(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        root.withdraw()
        try:
            canvas = tk.Canvas(root, width=900, height=400)
            preview = canvas.create_rectangle(0, 0, 900, 400, fill='white')
            fake = SimpleNamespace(polymer_qc_overlay_visible=True, polymer_qc={
                'accepted_contours': 3, 'candidate_components': 17, 'acceptance_percent': 100 * 3 / 17,
                'touches_scan_boundary': 1, 'branched_or_incomplete_skeleton': 12,
                'shorter_than_min_length': 1, 'longer_than_max_length': 0,
            })
            app.OrigamiCounterApp.draw_polymer_qc_overlay(fake, canvas, 900)
            items = canvas.find_withtag('qc_overlay')
            self.assertTrue(items)
            self.assertEqual(canvas.find_all()[0], preview)
            text = '\n'.join(canvas.itemcget(item, 'text') for item in items if canvas.type(item) == 'text')
            self.assertIn('Accepted 3 / 17', text)
            self.assertIn('topology 12', text)
            self.assertTrue(canvas.find_withtag('qc_details'))
            canvas.delete('qc_overlay')
            fake.polymer_qc_overlay_visible = False
            app.OrigamiCounterApp.draw_polymer_qc_overlay(fake, canvas, 900)
            self.assertFalse(canvas.find_withtag('qc_overlay'))
        finally:
            root.destroy()

    def test_topology_failure_not_mislabeled_as_excess_length(self):
        rgb = np.zeros((300, 300, 3), dtype=np.uint8)
        rgb[80:85, 60:240] = 255
        rgb[80:180, 148:153] = 255
        rgb[220:224, 60:120] = 255
        qc = {}
        with patch.object(app, 'scan_bbox', return_value=(0, 0, 300, 300)):
            objects = app.detect_polymer_contours(rgb, 900, 20, 0, 100, qc=qc)
        branches = [obj for obj in objects if obj.excluded_reason == 'branched_or_incomplete_skeleton']
        self.assertEqual(len(branches), 1)
        self.assertGreater(branches[0].branchpoint_pixels, 0)
        self.assertFalse(branches[0].path_valid)
        self.assertTrue(math.isnan(branches[0].length_nm))
        self.assertEqual(qc['longer_than_max_length'], 0)
        self.assertEqual(qc['accepted_contours'], 1)
        self.assertEqual(qc['candidate_components'], 2)
        self.assertEqual(qc['acceptance_percent'], 50)
        self.assertEqual(qc['polarity'], 'bright')
        self.assertTrue(0 < qc['foreground_percent'] < 100)
        self.assertEqual(qc['candidate_components'], qc['accepted_contours'] + sum(qc[reason] for reason in
            ('touches_scan_boundary', 'branched_or_incomplete_skeleton', 'shorter_than_min_length', 'longer_than_max_length')))

    def test_invalid_pixels_are_not_connected_by_artificial_lines(self):
        obj = app.PolymerObject(1, [(2, 2), (20, 20)], math.nan, math.nan, 0,
                                'branched_or_incomplete_skeleton', path_valid=False)
        image = Image.new('RGB', (24, 24))
        app.draw_polymer_trace(ImageDraw.Draw(image), obj, (255, 0, 0), 2)
        self.assertEqual(image.getpixel((10, 10)), (0, 0, 0))
        self.assertEqual(image.getpixel((2, 2)), (255, 0, 0))


if __name__ == '__main__':
    unittest.main()
