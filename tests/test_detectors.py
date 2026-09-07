"""
Unit tests for sequential change-point detectors (CUSUM, Page-Hinckley, EWMA) per Section 52.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.selection.detectors import (
    CUSUMDetector,
    PageHinckleyDetector,
    EWMADetector,
    create_detector,
)


def test_cusum_positive_detection():
    rng = np.random.default_rng(42)
    det = CUSUMDetector(delta=0.5, h=4.0, cooldown=5)

    # Stationary phase (mean = 0)
    events_pre = []
    for t in range(1, 50):
        val = float(rng.normal(0.0, 0.5))
        evt = det.update(val, current_round=t)
        if evt:
            events_pre.append(evt)
    assert len(events_pre) == 0, f"False alarm in stationary phase: {events_pre}"

    # Abrupt positive shift (mean = 2.0)
    events_post = []
    for t in range(50, 100):
        val = float(rng.normal(2.0, 0.5))
        evt = det.update(val, current_round=t)
        if evt:
            events_post.append(evt)

    assert len(events_post) >= 1, "CUSUM failed to detect positive shift"
    first_evt = events_post[0]
    assert first_evt.direction == "positive"
    assert first_evt.round >= 50
    delay = first_evt.round - 50
    print(f"CUSUM detected positive shift at t={first_evt.round} (delay={delay})")


def test_cusum_negative_detection():
    rng = np.random.default_rng(42)
    det = CUSUMDetector(delta=0.5, h=4.0, cooldown=5)

    for t in range(1, 50):
        val = float(rng.normal(0.0, 0.5))
        assert det.update(val, current_round=t) is None

    # Abrupt negative shift (mean = -2.0)
    events_post = []
    for t in range(50, 100):
        val = float(rng.normal(-2.0, 0.5))
        evt = det.update(val, current_round=t)
        if evt:
            events_post.append(evt)

    assert len(events_post) >= 1, "CUSUM failed to detect negative shift"
    assert events_post[0].direction == "negative"
    print(f"CUSUM detected negative shift at t={events_post[0].round}")


def test_page_hinckley():
    rng = np.random.default_rng(123)
    det = PageHinckleyDetector(delta=0.2, h=8.0, cooldown=5)

    # Stationary
    for t in range(1, 60):
        val = float(rng.normal(0.0, 0.5))
        evt = det.update(val, current_round=t)
        assert evt is None, f"False alarm at t={t}"

    # Shift at 60
    events = []
    for t in range(60, 120):
        val = float(rng.normal(1.5, 0.5))
        evt = det.update(val, current_round=t)
        if evt:
            events.append(evt)

    assert len(events) >= 1, "Page-Hinckley failed to detect shift"
    assert events[0].direction == "positive"
    print(f"Page-Hinckley detected shift at t={events[0].round}")


def test_ewma():
    rng = np.random.default_rng(456)
    det = EWMADetector(lambd=0.2, l_sigma=3.0, cooldown=5)

    for t in range(1, 50):
        val = float(rng.normal(0.0, 0.5))
        evt = det.update(val, current_round=t)
        assert evt is None, f"False alarm at t={t}"

    events = []
    for t in range(50, 100):
        val = float(rng.normal(2.0, 0.5))
        evt = det.update(val, current_round=t)
        if evt:
            events.append(evt)

    assert len(events) >= 1, "EWMA failed to detect shift"
    assert events[0].direction == "positive"
    print(f"EWMA detected shift at t={events[0].round}")


if __name__ == "__main__":
    test_cusum_positive_detection()
    test_cusum_negative_detection()
    test_page_hinckley()
    test_ewma()
    print("\nAll detector unit tests passed!")
