import tkinter as tk
import unittest

import numpy as np
from PIL import Image, ImageTk

import origami_counter_app as app


class PlotZoomTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.root.geometry('700x500')

    def tearDown(self):
        if hasattr(self, 'root'):
            self.root.destroy()

    def test_zoom_pan_fit_and_limits(self):
        frame = tk.Frame(self.root)
        frame.pack(fill='both', expand=True)
        canvas = tk.Canvas(frame)
        canvas.pack(fill='both', expand=True)
        source = Image.new('RGB', (1400, 1000))
        def draw(_event=None):
            image = app.zoomed_preview(source, canvas)
            canvas.photo = ImageTk.PhotoImage(image)
            canvas.delete('all')
            canvas.create_image(0, 0, image=canvas.photo, anchor='nw')
            canvas.configure(scrollregion=(0, 0, image.width, image.height))
        zoom = app.PreviewZoom(canvas, draw, self.root, before=frame)
        canvas.bind('<Configure>', draw)
        self.root.update()
        initial_width = canvas.photo.width()
        zoom.adjust(2)
        self.root.update()
        self.assertAlmostEqual(canvas.photo.width(), initial_width * 2, delta=1)
        canvas.scan_mark(300, 200)
        before = canvas.xview()
        canvas.scan_dragto(100, 200, gain=1)
        self.assertNotEqual(before, canvas.xview())
        zoom.reset()
        self.root.update()
        self.assertEqual(canvas.photo.width(), initial_width)
        zoom.adjust(0.0001)
        self.assertEqual(zoom.factor, 0.25)

    def test_all_polymer_panes_have_independent_zoom(self):
        ui = app.OrigamiCounterApp(self.root)
        source = Image.new('RGB', (1200, 800))
        ui.rgb = np.zeros((800, 1200, 3), dtype=np.uint8)
        ui.polymer_fit_plot_image = lambda: source
        ui.cached_polymer_figure_2b_image = lambda: source
        ui.cached_polymer_figure_c_image = lambda: source
        notebook = next(child for child in self.root.winfo_children() if isinstance(child, app.ttk.Notebook))
        notebook.select(notebook.tabs()[-1])
        for mode, canvas in [('analysis', ui.polymer_canvas), ('analysis', ui.polymer_plot_canvas),
                             ('figure_2b', ui.polymer_figure_2b_canvas), ('figure_c', ui.polymer_figure_c_canvas)]:
            with self.subTest(mode=mode, canvas=str(canvas)):
                ui.set_polymer_view(mode)
                self.root.update()
                zoom = canvas.preview_zoom
                zoom.adjust(2)
                self.root.update()
                self.assertEqual(zoom.factor, 2)
                self.assertTrue(canvas.bind('<MouseWheel>'))
                zoom.reset()
                self.assertEqual(zoom.factor, 1)


if __name__ == '__main__':
    unittest.main()
