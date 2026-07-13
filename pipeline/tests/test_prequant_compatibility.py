import json
from pathlib import Path

from pipeline import prequant_compatibility


class StubReport:
    def __init__(self, compatible: bool):
        self.compatible = compatible
        self.failures = () if compatible else (object(),)
        self.warnings = ()
        self.quantized_module_count = 3
        self.awq_mapping_count = 2

    def to_dict(self):
        return {"schema_version": 1, "compatible": self.compatible}


def test_main_writes_report_and_returns_zero_for_compatible(monkeypatch, tmp_path):
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        prequant_compatibility,
        "analyze_pipeline_config",
        lambda config, model_id=None: StubReport(True),
    )

    status = prequant_compatibility.main(
        ["--config", "recipe.yaml", "--output", str(output)]
    )

    assert status == 0
    assert json.loads(output.read_text()) == {
        "schema_version": 1,
        "compatible": True,
    }


def test_main_persists_failure_report_and_returns_two(monkeypatch, tmp_path):
    output = tmp_path / "nested" / "report.json"
    monkeypatch.setattr(
        prequant_compatibility,
        "analyze_pipeline_config",
        lambda config, model_id=None: StubReport(False),
    )

    status = prequant_compatibility.main(
        [
            "--config",
            "recipe.yaml",
            "--model-id",
            "local/model",
            "--output",
            str(output),
        ]
    )

    assert status == 2
    assert output.exists()
    assert json.loads(output.read_text())["compatible"] is False


def test_write_report_is_atomic(tmp_path):
    output = tmp_path / "report.json"
    prequant_compatibility.write_report(StubReport(True), output)

    assert output.exists()
    assert not Path(f"{output}.tmp").exists()
