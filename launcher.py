"""
UAE Law RAG — Executable Launcher
===================================
This is the PyInstaller entry point that boots the Streamlit web app
without requiring the `streamlit` CLI.

Directory layout when frozen (PyInstaller onedir):
    dist/UAE_Law_RAG/
    ├── UAE_Law_RAG.exe          ← this executable
    ├── db/                      ← vector database (user-visible)
    ├── logs/                    ← log files (user-visible)
    └── _internal/               ← PyInstaller runtime
        ├── app.py               ← Streamlit application
        ├── pagetree.py
        ├── hybrid_router.py
        ├── config.py
        └── ... (Python libs, DLLs, etc.)
"""

import os
import sys


def get_exe_dir():
    """Directory where the .exe lives (user-facing)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def get_internal_dir():
    """Directory with bundled Python files (_internal for PyInstaller)."""
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.dirname(sys.executable), "_internal")
    return os.path.dirname(os.path.abspath(__file__))


def setup_environment():
    """
    Configure paths so that:
      - CWD = exe directory (so db/, logs/ are next to the exe)
      - _internal/ is on sys.path (so app.py can import pagetree, etc.)
    """
    exe_dir = get_exe_dir()
    internal_dir = get_internal_dir()

    # Set working directory to exe location so db/ and logs/ are user-visible
    os.chdir(exe_dir)

    # Ensure _internal is on sys.path for module imports
    if internal_dir not in sys.path:
        sys.path.insert(0, internal_dir)
    if exe_dir not in sys.path:
        sys.path.insert(0, exe_dir)

    # Create runtime directories next to the exe
    os.makedirs(os.path.join(exe_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exe_dir, "db", "chroma"), exist_ok=True)


def main():
    setup_environment()

    app_script = os.path.join(get_internal_dir(), "app.py")

    if not os.path.exists(app_script):
        print(f"ERROR: Cannot find app.py at {app_script}")
        print("The application bundle may be corrupted.")
        input("Press Enter to exit...")
        sys.exit(1)

    print("=" * 60)
    print("  UAE Law RAG — Starting...")
    print("  Opening browser at http://localhost:8501")
    print("  Press Ctrl+C in this window to stop the server.")
    print("=" * 60)

    import streamlit.web.bootstrap as bootstrap

    # Disable file watcher and usage stats for frozen mode
    os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHERUSAGESTATS"] = "false"

    flag_options = {
        "server.port": 8501,
        "server.headless": True,
        "browser.gatherUsageStats": False,
        "global.developmentMode": False,
    }

    bootstrap.run(app_script, "", [], flag_options)


if __name__ == "__main__":
    main()