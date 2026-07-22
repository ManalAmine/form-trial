"""Compatibility entry point for collecting squat reps from a selected video."""

from __future__ import annotations

import sys

try:
    from .collect_squat_data import main
except ImportError:
    from collect_squat_data import main


if __name__ == "__main__":
    if "--source" not in sys.argv:
        raise SystemExit("Usage: python collect_squat_data_from_video.py --source path/to/video.mp4 [options]")
    main()
