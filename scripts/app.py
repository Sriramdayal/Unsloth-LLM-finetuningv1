"""Backward-compatible entry point for the Fine-Tuning Studio.

Moved to :mod:`src.ui.gradio_app`. Run with ``python scripts/app.py``
or the ``unsloth-gui`` console script.
"""

from src.ui.gradio_app import app, main  # noqa: F401

if __name__ == "__main__":
    main()
