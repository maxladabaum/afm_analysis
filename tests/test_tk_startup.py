import os
from pathlib import Path
import tempfile
import tkinter
import unittest
from unittest.mock import patch

import tk_startup


class TkStartupTests(unittest.TestCase):
    def test_windows_fallback_uses_base_python_with_unicode_path(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'Python installation 문서'
            tcl = base / 'tcl' / f'tcl{tkinter.TclVersion}'
            tk = base / 'tcl' / f'tk{tkinter.TkVersion}'
            tcl.mkdir(parents=True)
            tk.mkdir(parents=True)
            (tcl / 'init.tcl').touch()
            (tk / 'tk.tcl').touch()
            root = object()
            with patch.object(tk_startup.sys, 'platform', 'win32'), \
                 patch.object(tk_startup.sys, 'base_prefix', str(base)), \
                 patch.dict(os.environ, {}, clear=False), \
                 patch.object(tkinter, 'Tk', side_effect=[tkinter.TclError("Can't find a usable init.tcl"), root]) as create:
                self.assertIs(tk_startup.create_tk_root(), root)
                self.assertEqual(create.call_count, 2)
                self.assertEqual(os.environ['TCL_LIBRARY'], str(tcl))
                self.assertEqual(os.environ['TK_LIBRARY'], str(tk))

    def test_successful_startup_does_not_override_environment(self):
        root = object()
        with patch.dict(os.environ, {'TCL_LIBRARY': 'custom-tcl'}), patch.object(tkinter, 'Tk', return_value=root):
            self.assertIs(tk_startup.create_tk_root(), root)
            self.assertEqual(os.environ['TCL_LIBRARY'], 'custom-tcl')

    def test_missing_libraries_are_not_invented(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(tk_startup.sys, 'platform', 'win32'), \
             patch.object(tk_startup.sys, 'base_prefix', directory), \
             patch.object(tk_startup.sys, 'base_exec_prefix', directory), \
             patch.object(tk_startup.sys, '_base_executable', str(Path(directory) / 'python.exe')), \
             patch.object(tkinter, 'Tk', side_effect=tkinter.TclError('missing init.tcl')) as create:
            with self.assertRaisesRegex(tkinter.TclError, 'init.tcl'):
                tk_startup.create_tk_root()
            self.assertEqual(create.call_count, 1)

    def test_unrelated_tk_errors_are_not_retried(self):
        with patch.object(tk_startup.sys, 'platform', 'win32'), \
             patch.object(tkinter, 'Tk', side_effect=tkinter.TclError('unrelated error')) as create:
            with self.assertRaisesRegex(tkinter.TclError, 'unrelated'):
                tk_startup.create_tk_root()
            self.assertEqual(create.call_count, 1)


if __name__ == '__main__':
    unittest.main()
