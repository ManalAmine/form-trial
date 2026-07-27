"""Backward-compatible launcher for hybrid side-view squat prediction."""

try:
    from .predict_squat_hybrid_live import main
except ImportError:
    from predict_squat_hybrid_live import main


if __name__ == "__main__":
    main()
