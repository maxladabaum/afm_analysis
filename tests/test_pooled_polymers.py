import csv
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import origami_counter_app as app


class PooledPolymerTests(unittest.TestCase):
    def run_pool(self, missing=False):
        paths = [Path('first.jpg'), Path('second.jpg')]
        if missing:
            paths.append(Path('uncalibrated.jpg'))
        # Identical physical shapes at two pixel scales, reusing local contour ID 1.
        shape = [(0, 0), (20, 0), (20, 20), (40, 20)]
        def scale(rgb, path):
            if path.name == 'uncalibrated.jpg':
                return None
            return app.ScaleInfo(1000 if path.name == 'first.jpg' else 2000, 100, 0.1, True, 'spm')
        def detect(rgb, pixels_per_um, minimum, bias, maximum, **kwargs):
            self.assertEqual((minimum, bias, maximum), (10, 0, 100))
            kwargs['qc'].update(candidate_components=1, accepted_contours=1, acceptance_percent=100)
            factor = pixels_per_um / 1000
            return [app.PolymerObject(1, [(x * factor, y * factor) for x, y in shape], 60, 2000**0.5, 3)]
        with patch.object(app, 'load_rgb', return_value=np.zeros((100, 100, 3))), \
             patch.object(app, 'scale_info_from_spm', side_effect=scale), \
             patch.object(app, 'detect_polymer_contours', side_effect=detect):
            result = app.analyze_pooled_polymers(paths, {}, (10, 10, 0, 100, 0), lambda *_: None)
        return result, shape

    def test_scales_and_ids_and_pooled_fit(self):
        result, shape = self.run_pool()
        expected = app.polymer_msd_table([
            app.PolymerObject(1, shape, 60, 2000**0.5, 3),
            app.PolymerObject(2, shape, 60, 2000**0.5, 3),
        ], 1000, 10)
        for row, reference in zip(result.analysis.msd_rows, expected):
            for key, value in reference.items():
                self.assertEqual(row[key], value)
            self.assertEqual(row['image_ids'], '1;2')
            self.assertEqual(row['image_count'], 2)
        # Last separation has one pair per image: neither may be lost before pooling.
        self.assertEqual(result.analysis.msd_rows[-1]['sample_count'], 2)
        self.assertEqual(result.analysis.msd_rows[-1]['contour_separation_nm'], 60)
        self.assertEqual([row['pooled_id'] for row in result.contours], [1, 2])
        self.assertEqual([row['accepted_contours'] for row in result.images], [1, 1])
        self.assertTrue(math.isfinite(result.analysis.persistence_nm))

    def test_uncalibrated_images_are_reported_and_exported(self):
        result, _ = self.run_pool(missing=True)
        self.assertEqual(result.images[-1]['status'], 'skipped')
        self.assertIn('Box Scale', result.images[-1]['reason'])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            plot = app.OrigamiCounterApp.polymer_fit_plot_image(None, result.analysis, pooled=True)
            app.export_pooled_polymers(result, plot, output)
            self.assertEqual(len(list(output.iterdir())), 5)
            with (output / 'pooled_msd.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]['polymer_ids'], '1;2')
            with (output / 'images.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[-1]['status'], 'skipped')
            self.assertEqual(rows[0]['candidate_components'], '1')
            with Image.open(output / 'pooled_wlc_fit.png') as saved_plot:
                self.assertEqual(saved_plot.format, 'PNG')

    def test_saved_scale_and_no_fit(self):
        path = Path('saved.jpg')
        with patch.object(app, 'load_rgb', return_value=np.zeros((100, 100, 3))), \
             patch.object(app, 'scale_info_from_spm', return_value=None), \
             patch.object(app, 'detect_polymer_contours', return_value=[]) as detect:
            result = app.analyze_pooled_polymers([path], {app.image_key(path): {'pixels_per_um': 321}},
                                                (10, 10, 0, math.inf, 0), lambda *_: None)
        self.assertEqual(detect.call_args.args[1], 321)
        self.assertEqual(result.images[0]['scale_source'], 'saved')
        self.assertIsNone(result.analysis.persistence_nm)


if __name__ == '__main__':
    unittest.main()
