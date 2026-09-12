"""Compare work ASTs after removing only compression_phase contexts."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path

BASE = "a150f6902d7938f8b2b9d4e57be95e062a1be0c3"
TARGETS = {
    "pipeline/quantize.py": "_run_quantize",
    "src/llmcompressor/modifiers/transform/awq/base.py": "_apply_smoothing",
    "src/llmcompressor/pipelines/sequential/pipeline.py": "__call__",
}


class RemoveTimers(ast.NodeTransformer):
    def visit_With(self, node):
        node = self.generic_visit(node)
        node.items = [
            item
            for item in node.items
            if not (
                isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Name)
                and item.context_expr.func.id == "compression_phase"
            )
        ]
        return node if node.items else node.body


def normalized(source, name):
    module = ast.parse(source)
    candidates = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(candidates) == 1, (name, len(candidates))
    return ast.dump(RemoveTimers().visit(candidates[0]), include_attributes=False)


def main():
    rows = []
    for path, name in TARGETS.items():
        before = subprocess.check_output(["git", "show", f"{BASE}:{path}"], text=True)
        after = Path(path).read_text()
        equal = normalized(before, name) == normalized(after, name)
        rows.append(
            {
                "path": path,
                "function": name,
                "work_ast_identical": equal,
                "source_sha256": hashlib.sha256(after.encode()).hexdigest(),
            }
        )
    report = {
        "base": BASE,
        "scope": "Quantization work functions only; remove "
        "compression_phase wrappers and compare ASTs including operation order. "
        "Reporting integration is tested separately.",
        "checks": rows,
    }
    print(json.dumps(report, indent=2))
    assert all(row["work_ast_identical"] for row in rows), report


if __name__ == "__main__":
    main()
