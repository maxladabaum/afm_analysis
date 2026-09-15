import threading
import time
import tkinter as tk
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import origami_counter_app as app


class PolymerCalculationTests(unittest.TestCase):
    def test_worker_preserves_measurements_and_fit(self):
        polymer = app.PolymerObject(1, [(0, 0), (2, 0), (2, 2), (4, 2)], 6, 20**0.5, 3)
        expected_rows = app.polymer_msd_table([polymer], 1000, 1)
        expected_fit = app.fit_persistence_length_2d(
            np.array([row['contour_separation_nm'] for row in expected_rows]),
            np.array([row['mean_square_end_to_end_nm2'] for row in expected_rows]),
        )
        stages = []
        with patch.object(app, 'detect_polymer_contours', return_value=[polymer]):
            result = app.analyze_polymer_image(np.zeros((10, 10, 3)), 1000, 1, 1, 0,
                                              lambda value, text: stages.append(value))
        self.assertEqual(result.msd_rows, expected_rows)
        self.assertEqual((result.persistence_nm, result.fit_r2), expected_fit)
        self.assertEqual(stages, [25, 65, 78])

    def test_no_contours_finishes_without_fit(self):
        with patch.object(app, 'detect_polymer_contours', return_value=[]):
            result = app.analyze_polymer_image(np.zeros((10, 10, 3)), 1000, 1, 1, 0,
                                              lambda *_: None)
        self.assertIsNone(result.persistence_nm)
        self.assertEqual(result.objects, [])


class PolymerProgressTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f'Tk display unavailable: {exc}')
        self.root.withdraw()
        self.main_thread = threading.get_ident()
        self.updates = []
        self.fake_app = SimpleNamespace(root=self.root, polymer_progress_text=tk.StringVar(self.root))

        def update(value, text):
            self.assertEqual(threading.get_ident(), self.main_thread)
            self.updates.append((value, text))
            self.fake_app.polymer_progress_text.set(text)

        self.fake_app.update_polymer_progress = update

    def tearDown(self):
        if hasattr(self, 'root'):
            self.root.destroy()

    def run_worker(self):
        return app.OrigamiCounterApp.wait_for_polymer_analysis(
            self.fake_app, np.zeros((10, 10, 3)), 1000, 1, 1, 0,
        )

    def test_events_and_animation_continue_during_analysis(self):
        result = object()
        animation_values = []

        def heartbeat():
            for window in self.root.winfo_children():
                for widget in window.winfo_children():
                    if isinstance(widget, app.ttk.Progressbar):
                        animation_values.append(float(widget['value']))
            self.heartbeat_id = self.root.after(25, heartbeat)

        def calculate(*args, **kwargs):
            self.assertNotEqual(threading.get_ident(), self.main_thread)
            args[-1](25, 'Detecting contours')
            time.sleep(0.3)
            args[-1](78, 'Fitting')
            return result

        self.heartbeat_id = self.root.after(10, heartbeat)
        try:
            with patch.object(app, 'analyze_polymer_image', side_effect=calculate):
                self.assertIs(self.run_worker(), result)
        finally:
            self.root.after_cancel(self.heartbeat_id)
        self.assertGreater(len(set(animation_values)), 1)
        self.assertEqual(self.updates, [(25, 'Detecting contours'), (78, 'Fitting')])
        self.assertIsNone(self.root.grab_current())
        self.assertEqual(self.root.winfo_children(), [])

    def test_failure_closes_dialog_and_releases_grab(self):
        with patch.object(app, 'analyze_polymer_image', side_effect=ValueError('bad contour')):
            with self.assertRaisesRegex(RuntimeError, 'bad contour'):
                self.run_worker()
        self.assertIsNone(self.root.grab_current())
        self.assertEqual(self.root.winfo_children(), [])


if __name__ == '__main__':
    unittest.main()
