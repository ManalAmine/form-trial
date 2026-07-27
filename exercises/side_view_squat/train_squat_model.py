"""Backward-compatible launcher for the quality-only squat trainer."""

try:
    from .train_squat_quality_model import main
except ImportError:
    from train_squat_quality_model import main


if __name__ == "__main__":
    main()
