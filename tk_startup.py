"""Check Tk startup and recover Windows virtual-environment library discovery."""

import os
from pathlib import Path
import sys


def create_tk_root():
    import tkinter

    try:
        return tkinter.Tk()
    except tkinter.TclError as exc:
        if sys.platform != "win32" or not any(name in str(exc).lower() for name in ("init.tcl", "tk.tcl")):
            raise
        # Use only the libraries shipped with this interpreter, never a different
        # installed Python or an assumed version of Tcl/Tk.
        bases = [Path(sys.base_prefix), Path(sys.base_exec_prefix)]
        base_executable = getattr(sys, "_base_executable", None)
        if base_executable:
            bases.append(Path(base_executable).parent)
        for base in dict.fromkeys(bases):
            tcl = base / "tcl" / f"tcl{tkinter.TclVersion}"
            tk = base / "tcl" / f"tk{tkinter.TkVersion}"
            if not (tcl / "init.tcl").is_file() or not (tk / "tk.tcl").is_file():
                continue
            previous = {key: os.environ.get(key) for key in ("TCL_LIBRARY", "TK_LIBRARY")}
            os.environ["TCL_LIBRARY"] = str(tcl)
            os.environ["TK_LIBRARY"] = str(tk)
            try:
                return tkinter.Tk()
            except tkinter.TclError:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        raise


def check_tk() -> int:
    try:
        root = create_tk_root()
        try:
            root.withdraw()
            root.update_idletasks()
            print(f"Tkinter startup check passed (Tcl {root.tk.call('info', 'patchlevel')}).")
        finally:
            root.destroy()
        return 0
    except Exception as exc:
        error = exc
    print(f"Tkinter could not start: {error}", file=sys.stderr)
    print(f"Python executable: {sys.executable}\nBase installation: {sys.base_prefix}", file=sys.stderr)
    print(
        "Repair or update this Python installation using its python.org Windows installer.\n"
        "Use Modify to enable 'tcl/tk and IDLE', then Repair if that feature is already enabled.\n"
        "Run this launcher again after the repair. Reinstalling pip packages cannot repair Tcl/Tk.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(check_tk())
