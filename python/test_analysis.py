"""pytest: analysis/plot helpers operate on synthetic schema without GPU claims."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analyze_results import summarize


def test_summarize_empty():
    df = pd.DataFrame()
    text = summarize(df)
    assert "No rows" in text


def test_summarize_marks_unbenchmarked(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "operation": "vector_add",
                "variant": "cpu_ref",
                "size": 1024,
                "m": 0,
                "n": 0,
                "k": 0,
                "tile": 0,
                "median_ms": 0.5,
                "mean_ms": 0.5,
                "stddev_ms": 0.01,
                "metric_value": 1.0,
                "metric_unit": "GB/s",
                "correctness_passed": True,
                "status": "ok",
                "gpu_available": False,
            }
        ]
    )
    text = summarize(df)
    assert "UNBENCHMARKED" in text
    assert "vector_add" in text


def test_json_schema_roundtrip(tmp_path: Path):
    sample = {
        "operation": "vector_add",
        "variant": "cpu_ref",
        "kernel_timing": {"samples_ms": [1.0, 1.1], "median_ms": 1.05},
        "device": {"gpu_available": False, "name": "unavailable"},
        "status": "ok",
    }
    p = tmp_path / "sample.json"
    p.write_text(json.dumps(sample))
    loaded = json.loads(p.read_text())
    assert loaded["device"]["gpu_available"] is False
