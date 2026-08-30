"""Replay the rank-0 post-save sequence on a run that was killed after
``save_pretrained`` completed but before the side artifacts and gates ran.

WHY THIS EXISTS. On the Rancher cluster a kubelet flap marks the pod failed
immediately while containerd keeps the container running unsupervised; the
container is killed only when the kubelet returns. The GLM-5.3 full AWQ run
(job ``quant-glm52-awq-20260828t224453z``, 2026-08-29) lost its kubelet at
17:05:47Z, wrote a complete 394.6 GB checkpoint at 19:20, and was killed at
20:33:11Z when the kubelet came back -- so ~20.5 h of calibration and
compression survived, but everything ``quantize.py`` does after
``model.save_pretrained`` did not:

  * ``tokenizer.save_pretrained``            (quantize.py:1430)
  * ``_persist_ignore_to_config``            (quantize.py:1450)
  * ``versioning.write_recipe``              (quantize.py:1451)
  * ``assert_smooth_fold_consistency``       (quantize.py:1505)
  * ``assert_quant_checkpoint_verified``     (quantize.py:1511)
  * ``assert_no_ignore_shadowing``           (quantize.py:1532)

The first three are cheap writes. The last three are the ONLY numerical
validation the pipeline performs, so a checkpoint that skipped them is
structurally complete and entirely unvalidated -- which is the dangerous state,
because it looks finished. This module replays all six in the pipeline's order
so the artifact ends up indistinguishable from one whose run was not killed.

DELIBERATELY NOT A RESUME. It never loads model weights, never re-quantizes and
never rewrites a shard. If compression did not finish, this is the wrong tool:
check that the walk reached the last MoE layer first (the run's
``quant_metrics.rank-0.jsonl`` should mention the highest layer index) and
re-run the job instead.

ORDER MATTERS. ``assert_no_ignore_shadowing`` reads the FINAL config, so it must
run after ``_persist_ignore_to_config`` -- running it first tests a config the
pipeline would never have shipped. That is why this replays a sequence rather
than offering the steps a la carte.

Usage (from the repo root, with HF_HUB_CACHE pointed at the base model's cache):

    python -m pipeline.replay_post_save --run-dir /path/to/<timestamp> [--dry-run]

Config comes from the run's own ``config.yaml`` rather than a re-specified
recipe file, so the replay cannot drift from what actually ran.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pipeline import versioning
from pipeline.config import load_config
from pipeline.recipe import describe_recipe


def resolve_base_dir(model_id: str) -> Path | None:
    """Local directory holding the base model, or None.

    Reuses the run's own resolution helper so the replay reads the same
    snapshot the run did. A repo id like ``zai-org/GLM-5.3-BF16`` is not a path,
    which is exactly why the in-run dequant check silently skips on a
    repo-id config (quantize.py:1082 requires ``Path(base).is_dir()``); resolving
    it here is what makes the value-level check actually run.
    """
    from pipeline.quantize import _resolve_weight_index

    direct = Path(model_id)
    if (direct / "model.safetensors.index.json").is_file():
        return direct

    index = _resolve_weight_index(model_id)
    return index.parent if index is not None else None


def resolve_norm_gain_offset_offline(model_id: str, trust_remote_code: bool) -> float | None:
    """The architecture's norm gain form, without loading any weights.

    ``quantize.py:resolve_norm_gain_offset`` walks a live model's modules against
    the two registries in ``llmcompressor/preflight/quantization.py``. Those
    registries are the authority and this must not second-guess them, so rather
    than matching class names by hand we build the real module tree on the meta
    device (no weight bytes, no disk reads) and hand it to the same function.

    ``M3_NORM_GAIN_OFFSET`` overrides, for the case where remote code cannot be
    constructed offline. Returning None is safe: the smooth-fold gate skips with
    a printed reason rather than guessing, because with the wrong form a healthy
    fold reports a large error and fails the run at the very end.
    """
    override = os.environ.get("M3_NORM_GAIN_OFFSET", "").strip()
    if override:
        try:
            value = float(override)
        except ValueError:
            print(
                f"[replay] ignoring non-numeric M3_NORM_GAIN_OFFSET={override!r}",
                flush=True,
            )
        else:
            print(f"[replay] norm gain offset from env: {value:g}", flush=True)
            return value

    try:
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM

        from pipeline.quantize import resolve_norm_gain_offset

        config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        # Meta device: every module is constructed (so the registry walk sees the
        # real norm classes) but no parameter storage is allocated.
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=trust_remote_code
            )
        return resolve_norm_gain_offset(model)
    except Exception as err:  # noqa: BLE001 - a gate must not crash the replay
        print(
            f"[replay] could not resolve norm gain form offline "
            f"({type(err).__name__}: {err}); set M3_NORM_GAIN_OFFSET to enable "
            f"the smooth-fold gate",
            flush=True,
        )
        return None


def gate_layers_for(model_id: str, trust_remote_code: bool) -> list[int]:
    """The smooth-fold gate's layer list, derived from config, not a constant.

    Mirrors quantize.py:1466-1495 -- every MoE layer, plus the DSA indexer
    layers marked ``full`` in ``indexer_types``, since starting at
    ``first_k_dense_replace`` would leave the attention-side audit vacuous on
    GLM (layers 0-2 are the ones with their own indexer).
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    depth = int(getattr(config, "num_hidden_layers", 60))
    first_dense = int(getattr(config, "first_k_dense_replace", 3))
    layers = list(range(first_dense, depth))

    indexer_types = getattr(config, "indexer_types", None) or []
    indexer_layers = [i for i, kind in enumerate(indexer_types) if kind == "full"]
    added = sorted(set(indexer_layers) - set(layers))
    if added:
        layers = sorted(set(layers) | set(indexer_layers))
        print(
            f"[replay] smooth-fold gate: added indexer layers {added} so the "
            f"attention-side audit is not vacuous",
            flush=True,
        )
    return layers


def report_missing(ckpt: Path, run_dir: Path) -> list[str]:
    """Which post-save artifacts are absent. Presence-only, no validation."""
    missing = []
    if not (ckpt / "tokenizer_config.json").is_file():
        missing.append("tokenizer files")
    if not (run_dir / "recipe.json").is_file():
        missing.append("recipe.json")

    import json

    try:
        config = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
        ignore = config.get("quantization_config", {}).get("ignore", []) or []
        if not any(str(entry).startswith("re:") for entry in ignore):
            # _persist_ignore_to_config resolves patterns against saved tensors,
            # so a persisted list keeps the recipe's own `re:` entries for
            # modules compressed-tensors pruned (notably the MoE router).
            missing.append("persisted ignore patterns in config.json")
    except Exception as err:  # noqa: BLE001
        missing.append(f"config.json unreadable ({type(err).__name__}: {err})")

    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="the run directory containing config.yaml and checkpoint/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report which artifacts are missing and exit without writing or gating",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    ckpt = versioning.checkpoint_dir(run_dir)
    config_path = run_dir / "config.yaml"

    for path, what in ((config_path, "run config"), (ckpt, "checkpoint dir")):
        if not path.exists():
            print(f"[replay] FAIL: {what} not found at {path}", flush=True)
            return 2

    if not (ckpt / "model.safetensors.index.json").is_file():
        print(
            f"[replay] FAIL: {ckpt} has no model.safetensors.index.json, so "
            f"save_pretrained did not finish. This tool replays post-save steps "
            f"only -- re-run the quantization job instead.",
            flush=True,
        )
        return 2

    cfg = load_config(config_path)
    print(f"[replay] run dir   : {run_dir}")
    print(f"[replay] checkpoint: {ckpt}")
    print(f"[replay] model id  : {cfg.model.id}")
    print(f"[replay] recipe    : {cfg.quantization.method} / {cfg.quantization.scheme}")

    missing = report_missing(ckpt, run_dir)
    print(f"[replay] missing post-save artifacts: {missing or 'none'}", flush=True)
    if args.dry_run:
        return 0

    base_dir = resolve_base_dir(cfg.model.id)
    if base_dir is None:
        # Not fatal on its own: the structural half of the verify gate still
        # runs. But the dequant-vs-base check is the half that catches saved
        # garbage that happens to be finite, so say so loudly.
        print(
            f"[replay] WARNING: no local copy of {cfg.model.id} found, so the "
            f"dequant-vs-base and smooth-fold checks cannot run. Point "
            f"HF_HUB_CACHE at the base model's cache to enable them.",
            flush=True,
        )
    else:
        print(f"[replay] base dir  : {base_dir}", flush=True)

    # ---- Cheap writes, in quantize.py's order -----------------------------
    from pipeline.quantize import _persist_ignore_to_config

    print("\n[replay] === step 1/6: tokenizer ===", flush=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.id, trust_remote_code=cfg.model.trust_remote_code
    )
    tokenizer.save_pretrained(str(ckpt))
    print(f"[replay] wrote tokenizer to {ckpt}", flush=True)

    print("\n[replay] === step 2/6: persist ignore patterns ===", flush=True)
    _persist_ignore_to_config(ckpt, cfg.quantization.ignore)

    print("\n[replay] === step 3/6: recipe.json ===", flush=True)
    versioning.write_recipe(run_dir, describe_recipe(cfg.quantization))
    print(f"[replay] wrote {run_dir / 'recipe.json'}", flush=True)

    # ---- Gates. Fail-closed: the first failure raises and exits non-zero. --
    from pipeline.quantize import (
        assert_quant_checkpoint_verified,
        assert_smooth_fold_consistency,
    )
    from pipeline.serve_ignore import assert_no_ignore_shadowing

    print("\n[replay] === step 4/6: smooth-fold consistency gate ===", flush=True)
    if base_dir is None:
        print("[replay] SKIPPED (no local base model)", flush=True)
    else:
        offset = resolve_norm_gain_offset_offline(
            cfg.model.id, cfg.model.trust_remote_code
        )
        print(
            "[replay] norm gain form: "
            + ("unresolved" if offset is None else f"output * ({offset:g} + weight)"),
            flush=True,
        )
        layers = gate_layers_for(cfg.model.id, cfg.model.trust_remote_code)
        print(f"[replay] auditing {len(layers)} layers", flush=True)
        assert_smooth_fold_consistency(
            ckpt, base_dir, layers, norm_gain_offset=offset
        )

    print("\n[replay] === step 5/6: quant checkpoint verify gate ===", flush=True)
    assert_quant_checkpoint_verified(
        ckpt,
        base_dir,
        fp8_dynamic_targets=list(cfg.quantization.fp8_dynamic_targets or []),
    )

    print("\n[replay] === step 6/6: ignore-shadowing gate ===", flush=True)
    assert_no_ignore_shadowing(ckpt)

    print("\n[replay] ALL POST-SAVE STEPS COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
