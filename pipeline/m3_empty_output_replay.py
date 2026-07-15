"""Pure contracts for replaying one pinned MiniMax-M3 empty output."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPLAY_CAPS = (256, 16384)
EXPECTED_ATTEMPT = {
    "task": "mmlu_pro",
    "subtask": "mmlu_pro_economics",
    "doc_id": 45,
    "generation_seed": 1234,
}
EXPECTED_GENERATION = {
    "until": ["Question:"],
    "max_gen_toks": 256,
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "seed": 1234,
}
PINNED_CONFIG_GENERATION = {
    "temperature": 1.0,
    "top_p": 0.95,
    "do_sample": True,
    "max_gen_toks": 16384,
}
PINNED_LM_EVAL_VERSION = "0.4.12"


@dataclass(frozen=True)
class ReplayAttempt:
    """The validated source request for the exact empty-output replay."""

    attempt_uid: str
    prompt: str
    prompt_sha256: str
    generation_kwargs: dict[str, Any]
    source_row: dict[str, Any]


def _same_typed_value(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _same_typed_value(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_typed_value(item, expected_item)
            for item, expected_item in zip(actual, expected)
        )
    return actual == expected


def load_replay_attempt(path: Path, attempt_uid: str) -> ReplayAttempt:
    """Load and validate exactly one row matching the pinned replay request."""

    matches: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            if row.get("attempt_uid") == attempt_uid:
                matches.append(row)

    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one row for attempt_uid={attempt_uid!r}; "
            f"found {len(matches)}"
        )
    row = matches[0]

    for field, expected in EXPECTED_ATTEMPT.items():
        if not _same_typed_value(row.get(field), expected):
            raise ValueError(
                f"unexpected {field}: expected {expected!r}, got {row.get(field)!r}"
            )
    if row.get("response") != "":
        raise ValueError("replay source response must be exactly empty")

    arguments = row.get("generation_arguments")
    if not isinstance(arguments, list) or len(arguments) != 1:
        raise ValueError("generation_arguments must contain exactly one request")
    request = arguments[0]
    if not isinstance(request, list) or len(request) != 2:
        raise ValueError("generation_arguments request must be [prompt, kwargs]")
    prompt, generation_kwargs = request
    if not isinstance(prompt, str) or not isinstance(generation_kwargs, dict):
        raise ValueError("generation_arguments request must be [str, dict]")
    if not _same_typed_value(generation_kwargs, EXPECTED_GENERATION):
        raise ValueError(
            "generation kwargs do not match the pinned r4.5 smoke settings"
        )

    return ReplayAttempt(
        attempt_uid=attempt_uid,
        prompt=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        generation_kwargs=generation_kwargs,
        source_row=row,
    )


def postprocess_stages(
    raw_text: str,
    *,
    think_end_token: str,
    until: list[str],
) -> dict[str, Any]:
    """Expose the thinking-removal and sequential task-stop stages."""

    thinking_marker_present = bool(think_end_token) and think_end_token in raw_text
    after_thinking = (
        raw_text.split(think_end_token)[-1] if think_end_token else raw_text
    ).lstrip()
    after_task_stops = after_thinking
    matched_stop = None
    for stop in until:
        if stop and stop in after_task_stops:
            after_task_stops = after_task_stops.split(stop)[0]
            matched_stop = stop
    return {
        "raw_text": raw_text,
        "after_thinking": after_thinking,
        "after_task_stops": after_task_stops,
        "thinking_marker_present": thinking_marker_present,
        "matched_stop": matched_stop,
    }


def run_controls(
    attempt: ReplayAttempt,
    generate: Callable[[ReplayAttempt, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate both fixed-cap controls and retain each processing stage."""

    controls = []
    for cap in REPLAY_CAPS:
        completion = generate(attempt, cap)
        controls.append(
            {
                "max_gen_toks": cap,
                **completion,
                "postprocessing": postprocess_stages(
                    completion["raw_text"],
                    think_end_token="</mm:think>",
                    until=attempt.generation_kwargs["until"],
                ),
            }
        )
    return controls


def classify_controls(controls: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify where the fixed-cap replay becomes processed empty text."""

    by_cap = {control["max_gen_toks"]: control for control in controls}
    smoke = by_cap[256]
    production = by_cap[16384]
    smoke_stages = smoke["postprocessing"]
    production_stages = production["postprocessing"]
    smoke_empty = not smoke_stages["after_task_stops"].strip()
    production_empty = not production_stages["after_task_stops"].strip()
    if smoke["token_count"] == 0:
        kind = "zero_raw_tokens"
    elif smoke_empty and not production_empty and not smoke_stages["after_thinking"]:
        kind = "thinking_only_at_smoke_cap"
    elif smoke_empty and not production_empty and smoke_stages["matched_stop"]:
        kind = "task_stop_at_smoke_cap"
    elif (
        smoke_empty
        and smoke.get("finish_reason") == "length"
        and not production_empty
    ):
        kind = "length_cap_interaction"
    elif smoke_empty:
        kind = "processed_empty_unclassified"
    else:
        kind = "not_reproduced"
    return {
        "kind": kind,
        "smoke_processed_empty": smoke_empty,
        "production_processed_empty": production_empty,
    }


def build_replay_report(
    attempt: ReplayAttempt,
    *,
    config_path: Path,
    model_path: Path,
    generate: Callable[[ReplayAttempt, int], dict[str, Any]],
    versions: dict[str, str],
) -> dict[str, Any]:
    """Build the diagnostic sidecar without mutating source evaluation artifacts."""

    controls = run_controls(attempt, generate)
    source = attempt.source_row
    return {
        "schema_version": 1,
        "attempt_uid": attempt.attempt_uid,
        "task": source["task"],
        "subtask": source["subtask"],
        "doc_id": source["doc_id"],
        "generation_seed": source["generation_seed"],
        "prompt_sha256": attempt.prompt_sha256,
        "checkpoint_path": str(model_path),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "fixed_caps": list(REPLAY_CAPS),
        "versions": dict(versions),
        "controls": controls,
        "classification": classify_controls(controls),
    }


def write_replay_report(path: Path, report: dict[str, Any]) -> None:
    """Atomically write one JSON replay sidecar."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _distribution_version(
    version_getter: Callable[[str], str],
    *names: str,
) -> str:
    for name in names:
        try:
            return version_getter(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    raise importlib.metadata.PackageNotFoundError(names[-1])


def _validate_runtime_config(cfg: Any) -> None:
    if cfg.eval.backend != "vllm":
        raise ValueError("replay requires eval.backend='vllm'")
    if cfg.eval.enable_thinking is not True:
        raise ValueError("replay requires eval.enable_thinking=true")
    if cfg.eval.think_end_token != "</mm:think>":
        raise ValueError("replay requires eval.think_end_token='</mm:think>'")
    if not _same_typed_value(cfg.eval.gen_kwargs, PINNED_CONFIG_GENERATION):
        raise ValueError(
            "replay generation kwargs must match the pinned r4 sampling parameters"
        )


def _json_safe_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class _RawVllmGenerator:
    """Small callable adapter around the pinned lm-eval VLLM generation path."""

    def __init__(self, lm: Any, *, versions: dict[str, str]):
        self.lm = lm
        self.versions = versions
        self._closed = False

    def __call__(self, attempt: ReplayAttempt, cap: int) -> dict[str, Any]:
        from lm_eval.models.utils import maybe_truncate
        from vllm import SamplingParams

        eos = self.lm.tok_decode(self.lm.eot_token_id)
        kwargs = dict(attempt.generation_kwargs)
        kwargs["max_gen_toks"] = cap
        normalized, until, max_gen_toks = self.lm.modify_gen_kwargs(
            kwargs,
            eos=eos,
            default_max_gen_toks=self.lm.max_gen_toks,
        )
        token_ids = self.lm.tok_encode(attempt.prompt)
        token_ids, max_gen_toks = maybe_truncate(
            token_ids,
            max_gen_toks=max_gen_toks,
            max_model_len=self.lm.max_length,
            side=self.lm.truncation_side,
            verbose=True,
        )
        stop = [value for value in until if value == eos]
        params = SamplingParams(max_tokens=max_gen_toks, stop=stop, **normalized)
        request = self.lm._model_generate(
            requests=[token_ids],
            generate=True,
            sampling_params=[params],
        )[0]
        completion = request.outputs[0]
        completion_token_ids = [int(token) for token in completion.token_ids]
        return {
            "raw_text": str(completion.text),
            "token_ids": completion_token_ids,
            "token_count": len(completion_token_ids),
            "finish_reason": _json_safe_scalar(completion.finish_reason),
            "stop_reason": _json_safe_scalar(completion.stop_reason),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        cleanup = getattr(self.lm, "clean", None)
        if not callable(cleanup):
            cleanup = getattr(self.lm, "cleanup", None)
        if callable(cleanup):
            cleanup()


def load_raw_vllm_generator(
    config_path: Path,
    model_path: Path,
    *,
    model_loader: Callable[[Any, str], Any] | None = None,
    version_getter: Callable[[str], str] = importlib.metadata.version,
) -> _RawVllmGenerator:
    """Validate the pinned runtime before loading the model exactly once."""

    from pipeline.config import load_config

    lm_eval_version = _distribution_version(version_getter, "lm_eval", "lm-eval")
    if lm_eval_version != PINNED_LM_EVAL_VERSION:
        raise ValueError(
            "lm-eval version must be "
            f"{PINNED_LM_EVAL_VERSION}, received {lm_eval_version!r}"
        )
    cfg = load_config(config_path)
    _validate_runtime_config(cfg)
    versions = {
        "python": platform.python_version(),
        "lm_eval": lm_eval_version,
        "vllm": _distribution_version(version_getter, "vllm"),
    }
    if model_loader is None:
        from pipeline.lmeval_runner import _load_lm_model

        model_loader = _load_lm_model
    lm = model_loader(cfg, str(model_path))
    return _RawVllmGenerator(lm, versions=versions)


def run_raw_vllm_replay(
    attempt: ReplayAttempt,
    *,
    config_path: Path,
    model_path: Path,
    model_loader: Callable[[Any, str], Any] | None = None,
    version_getter: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, Any]:
    """Run both controls and always release the loaded backend."""

    generator = load_raw_vllm_generator(
        config_path,
        model_path,
        model_loader=model_loader,
        version_getter=version_getter,
    )
    try:
        return build_replay_report(
            attempt,
            config_path=config_path,
            model_path=model_path,
            generate=generator,
            versions=generator.versions,
        )
    finally:
        generator.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--attempt-uid", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    attempt = load_replay_attempt(args.samples, args.attempt_uid)
    report = run_raw_vllm_replay(
        attempt,
        config_path=args.config,
        model_path=args.model,
    )
    write_replay_report(args.out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
