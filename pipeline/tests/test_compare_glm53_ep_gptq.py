from __future__ import annotations

import numpy as np

try:
    from pipeline import compare_glm53_ep_gptq as comparison
except ImportError:
    comparison = None


def test_close_logits_pass_and_report_max_errors():
    assert comparison is not None, "logit comparison module is missing"
    reference = np.array([[1.0, -2.0, 4.0]], dtype=np.float32)
    candidates = {
        "ep4": reference + np.array([[1e-5, -2e-5, 0.0]], dtype=np.float32),
        "ep8": reference.copy(),
    }

    report = comparison.compare_logit_arrays(
        reference, candidates, rtol=2e-3, atol=2e-4
    )

    assert report["ok"] is True
    assert report["errors"] == []
    assert report["candidates"]["ep4"]["max_abs_error"] > 0


def test_nonfinite_or_out_of_tolerance_logits_fail():
    assert comparison is not None, "logit comparison module is missing"
    reference = np.array([[1.0, 2.0]], dtype=np.float32)
    candidates = {
        "far": np.array([[1.0, 2.1]], dtype=np.float32),
        "nan": np.array([[1.0, np.nan]], dtype=np.float32),
    }

    report = comparison.compare_logit_arrays(
        reference, candidates, rtol=2e-3, atol=2e-4
    )

    assert report["ok"] is False
    assert report["errors"] == [
        "far logits differ from DDP4 at rtol=0.002 atol=0.0002",
        "nan logits contain non-finite values",
    ]
