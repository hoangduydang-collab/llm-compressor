#!/usr/bin/env python
"""Persistently patch the installed vLLM to serve MiniMax-M3 W4AFP8 (W4A8 MoE).

Unlike ``pipeline/vllm_m3_patches.py`` (an in-process monkeypatch used by
``serve_verify``), this edits the vLLM source files in the active venv **once** so
that any launch path -- including the production ``vllm serve`` HTTP server --
works without a runtime hook.

Two edits (see BUGS_AND_FIXES.md "W4A8 MoE ... SWIGLUOAI_UNINTERLEAVE"):

  1. fused_moe/experts/cutlass_moe.py
     Add ``MoEActivation.SWIGLUOAI_UNINTERLEAVE`` to
     ``CutlassExpertsW4A8Fp8._supports_activation`` (the only tuple-form
     ``_supports_activation`` with exactly SILU/GELU/SWIGLUOAI).

  2. fused_moe/activation.py
     In ``apply_moe_activation``'s ``SWIGLUOAI_UNINTERLEAVE`` branch, default the
     clamp scalars to the M3/gpt-oss SwiGLU-OAI constants when the W4A8 call site
     passes none (it does), instead of asserting.

  3. model_executor/layers/fused_allreduce_gemma_rms_norm.py
     When CUDA graphs are enabled, skip FlashInfer fused AR in
     ``_can_use_flashinfer`` (NCCL fallback — graph-capturable). See BUGS_AND_FIXES.md
     "CUDA graph capture".

  4. model_executor/layers/fused_moe/router/base_router.py
     ``nan_to_num`` on ``router_logits`` in ``RouterBase._select_experts`` (the
     template method every router subclass funnels through, right before
     ``_compute_routing``; padding NaNs → duplicate/OOB expert IDs → W4A8 MoE IMA;
     vLLM #39288 / #39391).

     NOTE: this replaced an earlier edit to ``MoERunner._apply_quant_method`` — a
     **dead path** for M3 W4AFP8 (it uses the modular ``FusedMoEModularKernel`` /
     ``router/*``, not ``MoERunner``), which is why the IMA at capture 16/51 never
     moved despite that patch verifying as applied. See BUGS_AND_FIXES.md
     "CUDA graph capture".

Idempotent: re-running is a no-op. Fails loudly if the expected code is not found
(so a vLLM upgrade that changes these files can't silently leave a broken serve).

Usage:
    python pipeline/slurm/patch_vllm_m3_serve.py            # apply
    python pipeline/slurm/patch_vllm_m3_serve.py --check    # report only, exit 1 if unpatched

Removal criteria: delete this script and revert once a vLLM release serves M3
W4A8 (SwiGLU-OAI uninterleaved) natively.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# M3 / gpt-oss SwiGLU-OAI constants: gate*sigmoid(alpha*gate)*(up+beta), clamped.
SWIGLU_LIMIT = 7.0
SWIGLU_ALPHA = 1.702
SWIGLU_BETA = 1.0

_MARK = "llmc M3 W4A8 SWIGLUOAI_UNINTERLEAVE patch"
_CG_AR_MARK = "llmc M3 cudagraph: skip FlashInfer fused AR"
_CG_MOE_MARK = "llmc M3 cudagraph: nan_to_num router_logits in _select_experts"
_PROBE_MARK = "llmc M3 MoE quality probe"
_LOAD_AUDIT_MARK = "llmc M3 load audit"

# Optional, env-gated (M3_MOE_PROBE=1) diagnostic appended to the vLLM M3 model
# module so it runs inside the spawned Worker_TP* processes (in-process
# monkeypatches in serve_verify do NOT reach workers). It confirms/rules out the
# serve-side "arring" garbage root cause (shared expert dropped in every MoE
# layer) by logging, for the first few real-prefill MoE forwards, the shared-
# expert output norm and the combined MoE output norm. shared_norm~=0 / missing
# module (or, with M3_MOE_PROBE_RECOMPUTE=1, moe_out ~= routed*2) => the shared
# expert is dropped -> residual collapse -> garbage (aquaman164
# m3_official_loader.py "M3_FIX_SHARED"; see BUGS_AND_FIXES.md "full-calib AWQ
# garbage output"). No runtime cost unless M3_MOE_PROBE=1.
_PROBE_BLOCK = '''

# === {mark} (BUGS_AND_FIXES.md "full-calib AWQ garbage output") ===
try:  # gated diagnostic; never break model import
    import os as _llmc_os
    if _llmc_os.environ.get("M3_MOE_PROBE") == "1":
        from vllm.logger import init_logger as _llmc_init_logger
        _llmc_probe_log = _llmc_init_logger("llmc.m3_moe_probe")
        _llmc_probe_moe_cls = globals().get("MiniMaxM3MoE")
        _llmc_probe_state = {{"n": 0, "nf": 0}}
        _llmc_probe_max = int(_llmc_os.environ.get("M3_MOE_PROBE_LAYERS", "6"))
        _llmc_probe_nf_max = int(_llmc_os.environ.get("M3_MOE_PROBE_NF", "4"))
        _llmc_probe_recompute = _llmc_os.environ.get("M3_MOE_PROBE_RECOMPUTE") == "1"

        import math as _llmc_math

        def _llmc_probe_norm(_t):
            try:
                return float(_t.float().norm().item())
            except Exception:
                return -1.0

        if _llmc_probe_moe_cls is not None and not getattr(
            _llmc_probe_moe_cls, "_llmc_probed", False
        ):
            _llmc_probe_orig_forward = _llmc_probe_moe_cls.forward

            import torch as _llmc_torch

            def _llmc_probe_forward(self, hidden_states, *args, **kwargs):
                out = _llmc_probe_orig_forward(self, hidden_states, *args, **kwargs)
                try:
                    # HARD GATE: never touch the tensor while a CUDA graph is being
                    # captured. Any .item()/.norm().item() forces a device sync, which
                    # is ILLEGAL during capture and poisons the whole capture stream
                    # ("operation failed due to a previous error during capture" ->
                    # every Worker_TP* dies). The probe only cares about REAL prefills,
                    # so skip capture (and profiling dummy) passes entirely, up front,
                    # BEFORE computing any norm.
                    if (
                        _llmc_probe_state["n"] >= _llmc_probe_max
                        or _llmc_torch.cuda.is_current_stream_capturing()
                    ):
                        return out
                    n = int(hidden_states.shape[0])
                    hs = hidden_states.view(-1, hidden_states.shape[-1])
                    in_norm = _llmc_probe_norm(hs)
                    # Eager PROFILING/WARMUP dummy runs: vLLM's pre-capture warmup feeds
                    # uninitialized/dummy input whose norm is ~0 (zeros) or NaN/inf
                    # (uninit memory). NaN slips past a "<= 1e-6" test (all NaN compares
                    # are False), which previously let warmup eat the whole probe budget
                    # before the real prompt. So DON'T spend the main (finite-input)
                    # budget on it. But a non-finite input on a REAL prefill would itself
                    # be the garbage root cause (NaN propagating from upstream:
                    # embed/attn/norm/indexer), so surface a bounded number of
                    # non-finite hits separately instead of hiding them entirely.
                    if not _llmc_math.isfinite(in_norm):
                        if _llmc_probe_state["nf"] < _llmc_probe_nf_max:
                            _llmc_probe_state["nf"] += 1
                            _llmc_probe_log.warning(
                                "M3_MOE_PROBE_NONFINITE#%d tokens=%d in_norm=%s "
                                "(warmup/dummy OR real upstream NaN -> garbage; if these "
                                "keep firing AFTER 'Capturing CUDA graphs' completes on "
                                "the real prompt, NaN is propagating from upstream)",
                                _llmc_probe_state["nf"], n, in_norm,
                            )
                        return out
                    if in_norm <= 1e-6:
                        return out
                    if _llmc_probe_state["n"] < _llmc_probe_max and 2 <= n <= 64:
                        _llmc_probe_state["n"] += 1
                        shared_mod = getattr(self, "shared_experts", None)
                        shared_norm = (
                            _llmc_probe_norm(shared_mod(hs))
                            if shared_mod is not None else -1.0
                        )
                        out_norm = _llmc_probe_norm(out)
                        routed_norm = -1.0
                        ratio = -1.0
                        if _llmc_probe_recompute:
                            _rl, _ = self.gate(hs)
                            _routed = self.experts(hidden_states=hs, router_logits=_rl)
                            routed_norm = _llmc_probe_norm(_routed)
                            ratio = out_norm / routed_norm if routed_norm > 0 else -1.0
                        # Real input (in_norm>0): shared missing/zero => zero-loaded
                        # shared expert; moe_out ~= routed*2 (recompute) => runner
                        # never added shared. Either => dropped in every MoE layer.
                        dropped = in_norm > 1e-6 and (
                            shared_mod is None
                            or (0.0 <= shared_norm <= 1e-3)
                            or (0.0 <= ratio and abs(ratio - 2.0) < 0.05)
                        )
                        _llmc_probe_log.warning(
                            "M3_MOE_PROBE#%d tokens=%d in_norm=%.3f shared_present=%s "
                            "shared_norm=%.3f moe_out_norm=%.3f routed_norm=%.3f "
                            "out/routed=%.3f%s",
                            _llmc_probe_state["n"], n, in_norm, shared_mod is not None,
                            shared_norm, out_norm, routed_norm, ratio,
                            "  <-- SHARED EXPERT DROPPED (garbage root cause)"
                            if dropped else "  (shared expert contributing)",
                        )
                except Exception as _llmc_e:
                    _llmc_probe_log.warning("M3_MOE_PROBE forward failed: %r", _llmc_e)
                return out

            _llmc_probe_moe_cls.forward = _llmc_probe_forward
            _llmc_probe_moe_cls._llmc_probed = True
            _llmc_probe_log.warning(
                "llmc M3 MoE probe active on %s.forward (M3_MOE_PROBE=1, "
                "recompute=%s, max_layers=%d); skips capture + zero/non-finite warmup runs",
                _llmc_probe_moe_cls.__name__, _llmc_probe_recompute, _llmc_probe_max,
            )
except Exception:
    pass
# === end {mark} ===
'''.format(mark=_PROBE_MARK)

# Optional, env-gated loader audit. The M3 VL top-level model delegates
# language-model weights to MiniMaxM3Model.load_weights(), whose routed-expert
# mapping historically recognized only w1/w2/w3. This hook records which
# checkpoint routed-expert, shared-expert, and lm-head tensors actually reach a
# parameter weight_loader without retaining tensor data or changing the mapping.
_LOAD_AUDIT_BLOCK = r'''

# === llmc M3 load audit (MiniMax-M3 routed-expert wiring) ===
try:  # gated diagnostic; never break model import
    import os as _llmc_audit_os
    _llmc_audit_enabled = _llmc_audit_os.environ.get("M3_LOAD_AUDIT") == "1"
    _llmc_fp_enabled = (
        _llmc_audit_os.environ.get("M3_PARAM_FINGERPRINT") == "1"
    )
    if _llmc_audit_enabled or _llmc_fp_enabled:
        import hashlib as _llmc_fp_hashlib
        import json as _llmc_fp_json
        import re as _llmc_fp_re

        import torch as _llmc_fp_torch
        from vllm.logger import init_logger as _llmc_audit_init_logger
        _llmc_audit_log = _llmc_audit_init_logger("llmc.m3_load_audit")
        _llmc_fp_max_samples = 256
        _llmc_fp_layers = {
            int(_item)
            for _item in _llmc_audit_os.environ.get(
                "M3_PARAM_FINGERPRINT_LAYERS", "3,59"
            ).split(",")
            if _item.strip().isdigit()
        }
        _llmc_fp_case = _llmc_audit_os.environ.get(
            "M3_QUALITY_CASE", "unspecified"
        )
        _llmc_audit_cls = globals().get(
            "MiniMaxM3SparseForConditionalGeneration"
        )

        def _llmc_audit_tracked_name(_name):
            return (
                "block_sparse_moe.experts." in _name
                or "block_sparse_moe.shared_experts." in _name
                or "lm_head" in _name
            )

        def _llmc_audit_projection(_name):
            for _alias, _label in (
                (".gate_proj.", "gate_proj"),
                (".up_proj.", "up_proj"),
                (".down_proj.", "down_proj"),
                (".w1.", "w1"),
                (".w2.", "w2"),
                (".w3.", "w3"),
            ):
                if _alias in _name:
                    return _label
            if "shared_experts" in _name:
                return "shared"
            if "lm_head" in _name:
                return "lm_head"
            return "other"

        def _llmc_fp_category(_name):
            _lower = _name.lower()
            _match = _llmc_fp_re.search(r"[.]layers[.](\d+)[.]", _lower)
            _layer = int(_match.group(1)) if _match else None
            if "lm_head" in _lower:
                return "lm_head", _layer
            if _layer not in _llmc_fp_layers:
                return None, _layer
            if "shared_experts" in _lower:
                return "shared_expert", _layer
            if ".experts." in _lower:
                return "routed_expert", _layer
            if "indexer" in _lower:
                return "msa_indexer", _layer
            if any(
                _part in _lower
                for _part in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "qkv_proj",
                    "qkv.",
                )
            ):
                return "attention_qkv", _layer
            return None, _layer

        def _llmc_emit_fingerprints(_model, _scope):
            if not _llmc_fp_enabled:
                return
            _found = set()
            _errors = []
            try:
                _dist = getattr(_llmc_fp_torch, "distributed", None)
                _rank = (
                    int(_dist.get_rank())
                    if _dist is not None
                    and _dist.is_available()
                    and _dist.is_initialized()
                    else int(_llmc_audit_os.environ.get("RANK", "0"))
                )
            except Exception:
                _rank = -1
            for _name, _param in _model.named_parameters():
                _category, _layer = _llmc_fp_category(_name)
                if _category is None:
                    continue
                _found.add(_category)
                try:
                    _flat = _param.detach().reshape(-1)
                    _numel = int(_flat.numel())
                    _sample_count = min(_llmc_fp_max_samples, _numel)
                    if _sample_count:
                        _indices = _llmc_fp_torch.linspace(
                            0,
                            _numel - 1,
                            steps=_sample_count,
                            device=_flat.device,
                        ).long()
                        _sample = _flat.index_select(0, _indices)
                        _sample_cpu = _sample.detach().cpu().contiguous()
                        _digest = _llmc_fp_hashlib.sha256(
                            _sample_cpu.view(_llmc_fp_torch.uint8).numpy().tobytes()
                        ).hexdigest()
                    else:
                        _sample_cpu = _flat.detach().cpu()
                        _digest = _llmc_fp_hashlib.sha256(b"").hexdigest()
                    _record = {
                        "case": _llmc_fp_case,
                        "scope": _scope,
                        "rank": _rank,
                        "name": _name,
                        "category": _category,
                        "layer": _layer,
                        "dtype": str(_param.dtype),
                        "shape": list(_param.shape),
                        "numel": _numel,
                        "sample_count": _sample_count,
                        "sample_sha256": _digest,
                    }
                    if _sample_cpu.is_floating_point():
                        _float = _sample_cpu.float()
                        _finite = _llmc_fp_torch.isfinite(_float)
                        _finite_values = _float[_finite]
                        _record["finite_fraction"] = float(
                            _finite.float().mean().item()
                        )
                        if int(_finite_values.numel()):
                            _record.update(
                                {
                                    "sample_abs_max": float(
                                        _finite_values.abs().max().item()
                                    ),
                                    "sample_mean": float(
                                        _finite_values.mean().item()
                                    ),
                                    "sample_std": float(
                                        _finite_values.std(unbiased=False).item()
                                    ),
                                }
                            )
                        if _numel <= 1_000_000:
                            _record["full_norm"] = float(
                                _flat.float().norm().item()
                            )
                    _llmc_audit_log.warning(
                        "M3_PARAM_FINGERPRINT# %s",
                        _llmc_fp_json.dumps(_record, sort_keys=True),
                    )
                except Exception as _llmc_fp_exc:
                    _errors.append({"name": _name, "error": repr(_llmc_fp_exc)})
            _expected = {
                "lm_head",
                "shared_expert",
                "routed_expert",
                "attention_qkv",
                "msa_indexer",
            }
            _summary = {
                "case": _llmc_fp_case,
                "scope": _scope,
                "found": sorted(_found),
                "missing": sorted(_expected - _found),
                "errors": _errors,
            }
            _llmc_audit_log.warning(
                "M3_PARAM_FINGERPRINT_SUMMARY# %s",
                _llmc_fp_json.dumps(_summary, sort_keys=True),
            )

        if _llmc_audit_cls is not None and not getattr(
            _llmc_audit_cls, "_llmc_load_audited", False
        ):
            _llmc_audit_orig_load_weights = _llmc_audit_cls.load_weights

            def _llmc_audit_load_weights(self, weights):
                _llmc_state = {
                    "current": None,
                    "seen": [],
                    "matches": [],
                }
                _llmc_restore = []

                # AutoWeightsLoader delegates nested M3 weights to the custom
                # language-model loader. Wrapping these parameters records the
                # actual source-key -> target-parameter handoff across both
                # loader layers, including default-weight-loader lm_head tensors.
                try:
                    for _target, _param in self.named_parameters():
                        if not _llmc_audit_tracked_name(_target):
                            continue
                        _had_loader = hasattr(_param, "weight_loader")
                        _original_loader = getattr(_param, "weight_loader", None)
                        _base_loader = (
                            _original_loader
                            if callable(_original_loader)
                            else default_weight_loader
                        )

                        def _llmc_audit_weight_loader(
                            *args,
                            _target_name=_target,
                            _base=_base_loader,
                            **kwargs
                        ):
                            _source_name = _llmc_state["current"]
                            if _source_name is not None:
                                _llmc_state["matches"].append(
                                    (_source_name, _target_name)
                                )
                            return _base(*args, **kwargs)

                        _llmc_restore.append(
                            (_param, _had_loader, _original_loader)
                        )
                        _param.weight_loader = _llmc_audit_weight_loader

                    def _llmc_audit_weights():
                        for _source_name, _weight in weights:
                            if _llmc_audit_tracked_name(_source_name):
                                _llmc_state["current"] = _source_name
                                _llmc_state["seen"].append(_source_name)
                                yield _source_name, _weight
                                _llmc_state["current"] = None
                            else:
                                yield _source_name, _weight

                    return _llmc_audit_orig_load_weights(
                        self, _llmc_audit_weights()
                    )
                finally:
                    for _param, _had_loader, _original_loader in _llmc_restore:
                        if _had_loader:
                            _param.weight_loader = _original_loader
                        else:
                            delattr(_param, "weight_loader")

                    _llmc_seen = _llmc_state["seen"]
                    _llmc_matched = {source for source, _ in _llmc_state["matches"]}
                    _llmc_unmatched = [
                        source for source in _llmc_seen
                        if source not in _llmc_matched
                    ]
                    _llmc_counts = {}
                    for _source in _llmc_seen:
                        _kind = _llmc_audit_projection(_source)
                        _llmc_counts[_kind] = _llmc_counts.get(_kind, 0) + 1
                    _llmc_sample = "; ".join(
                        "%s -> %s" % (_source, _target)
                        for _source, _target in _llmc_state["matches"][:8]
                    )
                    _llmc_unmatched_sample = "; ".join(_llmc_unmatched[:8])
                    _llmc_audit_log.warning(
                        "M3_LOAD_AUDIT# seen=%d matched=%d unmatched_this_rank=%d "
                        "by_projection=%s sample=%s unmatched_sample=%s",
                        len(_llmc_seen),
                        len(_llmc_matched),
                        len(_llmc_unmatched),
                        _llmc_counts,
                        _llmc_sample or "NONE",
                        _llmc_unmatched_sample or "NONE",
                    )
                    _llmc_emit_fingerprints(
                        self, "MiniMaxM3SparseForConditionalGeneration"
                    )

            _llmc_audit_cls.load_weights = _llmc_audit_load_weights
            _llmc_audit_cls._llmc_load_audited = True
            _llmc_audit_log.warning(
                "llmc M3 load audit active "
                "(M3_LOAD_AUDIT=1; no tensors retained or remapped)"
            )

        # The top-level AutoWeightsLoader does not expose the custom M3
        # expert-alias decision. Instrument that decision point directly: it
        # compares checkpoint aliases against get_expert_mapping() before it
        # invokes the fused-MoE parameter weight loader.
        _llmc_audit_model_cls = globals().get("MiniMaxM3Model")
        if _llmc_audit_model_cls is not None and not getattr(
            _llmc_audit_model_cls, "_llmc_load_audited", False
        ):
            _llmc_audit_orig_model_load_weights = (
                _llmc_audit_model_cls.load_weights
            )

            def _llmc_audit_model_load_weights(self, weights):
                _llmc_mapping_aliases = sorted({
                    _weight_name
                    for _, _weight_name, _, _ in self.get_expert_mapping()
                })
                _llmc_state = {
                    "current": None,
                    "seen": [],
                    "matches": [],
                }
                _llmc_restore = []
                try:
                    for _target, _param in self.named_parameters():
                        if (
                            "block_sparse_moe.experts." not in _target
                            and "block_sparse_moe.shared_experts." not in _target
                        ):
                            continue
                        _had_loader = hasattr(_param, "weight_loader")
                        _original_loader = getattr(_param, "weight_loader", None)
                        _base_loader = (
                            _original_loader
                            if callable(_original_loader)
                            else default_weight_loader
                        )

                        def _llmc_audit_model_weight_loader(
                            *args,
                            _target_name=_target,
                            _base=_base_loader,
                            **kwargs
                        ):
                            _source_name = _llmc_state["current"]
                            if _source_name is not None:
                                _llmc_state["matches"].append(
                                    (_source_name, _target_name)
                                )
                            return _base(*args, **kwargs)

                        _llmc_restore.append(
                            (_param, _had_loader, _original_loader)
                        )
                        _param.weight_loader = _llmc_audit_model_weight_loader

                    def _llmc_audit_model_weights():
                        for _source_name, _weight in weights:
                            if _llmc_audit_tracked_name(_source_name):
                                _llmc_state["current"] = _source_name
                                _llmc_state["seen"].append(_source_name)
                                yield _source_name, _weight
                                _llmc_state["current"] = None
                            else:
                                yield _source_name, _weight

                    return _llmc_audit_orig_model_load_weights(
                        self, _llmc_audit_model_weights()
                    )
                finally:
                    for _param, _had_loader, _original_loader in _llmc_restore:
                        if _had_loader:
                            _param.weight_loader = _original_loader
                        else:
                            delattr(_param, "weight_loader")

                    _llmc_seen = _llmc_state["seen"]
                    _llmc_matched = {
                        _source for _source, _ in _llmc_state["matches"]
                    }
                    _llmc_unsupported_routed = [
                        _source
                        for _source in _llmc_seen
                        if (
                            "block_sparse_moe.experts." in _source
                            and not any(
                                f".{_alias}." in _source
                                for _alias in _llmc_mapping_aliases
                            )
                        )
                    ]
                    _llmc_shared_seen = sum(
                        "block_sparse_moe.shared_experts." in _source
                        for _source in _llmc_seen
                    )
                    _llmc_sample = "; ".join(
                        "%s -> %s" % (_source, _target)
                        for _source, _target in _llmc_state["matches"][:8]
                    )
                    _llmc_audit_log.warning(
                        "M3_LOAD_AUDIT# scope=model mapping_aliases=%s seen=%d "
                        "matched=%d unmatched_this_scope=%d unsupported_routed=%d "
                        "shared_seen=%d sample=%s unsupported_sample=%s",
                        _llmc_mapping_aliases,
                        len(_llmc_seen),
                        len(_llmc_matched),
                        len(_llmc_seen) - len(_llmc_matched),
                        len(_llmc_unsupported_routed),
                        _llmc_shared_seen,
                        _llmc_sample or "NONE",
                        "; ".join(_llmc_unsupported_routed[:8]) or "NONE",
                    )
                    _llmc_emit_fingerprints(self, "MiniMaxM3Model")

            _llmc_audit_model_cls.load_weights = _llmc_audit_model_load_weights
            _llmc_audit_model_cls._llmc_load_audited = True
            _llmc_audit_log.warning(
                "llmc M3 direct loader audit active "
                "(reports checkpoint aliases and unsupported routed tensors)"
            )
except Exception:
    pass
# === end llmc M3 load audit ===
'''


def _vllm_dir() -> Path:
    import vllm

    return Path(vllm.__file__).resolve().parent


# FlashInfer >= 0.6.10 restored the finalizeMoeRoutingKernel bounds check that was
# dropped in 0.5.3 (flashinfer#2762). Missing it => padding tokens during CUDA
# graph capture index out-of-bounds in the MoE finalize -> deterministic IMA
# (vLLM #35706 / #42906). This is a *separate* suspect from the router NaN patch:
# it affects the flashinfer-backed MoE finalize, not vLLM's native W4A8 grouped
# GEMM. Report it so a stale quant-venv flashinfer is caught immediately.
_FLASHINFER_MIN_SAFE = (0, 6, 10)


def _report_flashinfer_version() -> None:
    try:
        import flashinfer  # type: ignore

        ver = getattr(flashinfer, "__version__", "?")
        print(f"flashinfer {ver}")
        parts = re.findall(r"\d+", str(ver))[:3]
        if len(parts) == 3:
            tup = tuple(int(p) for p in parts)
            if tup < _FLASHINFER_MIN_SAFE:
                print(
                    f"  WARNING: flashinfer {ver} < 0.6.10 lacks the "
                    "finalizeMoeRoutingKernel bounds-check fix (flashinfer#2762 / "
                    "vLLM #42906). If the CUDA-graph IMA is in a flashinfer MoE "
                    "finalize (confirm with CUDA_LAUNCH_BLOCKING=1), upgrade: "
                    '"$UV" pip install -U "flashinfer-python>=0.6.11.post2"'
                )
    except Exception as exc:  # noqa: BLE001
        print(f"flashinfer: not importable ({exc})")


def _patch_supports_activation(text: str) -> tuple[str, bool, bool]:
    """Add SWIGLUOAI_UNINTERLEAVE to the W4A8 tuple-form _supports_activation.

    Returns (new_text, changed, found).
    """
    # The W4A8 kernel is the ONLY class using a tuple (parentheses) with exactly
    # these three members; every other _supports_activation uses a list.
    pattern = re.compile(
        r"(?P<head>return\s+activation\s+in\s+\(\s*\n"
        r"(?P<ind>[ \t]+)MoEActivation\.SILU,[ \t]*\n"
        r"[ \t]+MoEActivation\.GELU,[ \t]*\n"
        r"[ \t]+MoEActivation\.SWIGLUOAI,[ \t]*\n)"
        r"(?P<close>[ \t]*\))",
        re.MULTILINE,
    )
    m = pattern.search(text)
    if m is None:
        # Either already patched (enum present) or layout changed.
        if "MoEActivation.SWIGLUOAI_UNINTERLEAVE" in text:
            return text, False, True
        return text, False, False

    ind = m.group("ind")
    injected = (
        m.group("head")
        + f"{ind}MoEActivation.SWIGLUOAI_UNINTERLEAVE,\n"
        + m.group("close")
    )
    return text[: m.start()] + injected + text[m.end() :], True, True


def _patch_apply_activation(text: str) -> tuple[str, bool, bool]:
    """Replace the SWIGLUOAI_UNINTERLEAVE assert with a clamp-scalar default."""
    assert_line = re.compile(
        r"^(?P<indent>[ \t]+)assert clamp_limit is not None,"
        r'\s*"SWIGLUOAI_UNINTERLEAVE requires clamp_limit"\s*$',
        re.MULTILINE,
    )
    if _MARK in text:
        return text, False, True

    m = assert_line.search(text)
    if m is None:
        return text, False, False

    indent = m.group("indent")
    replacement = (
        f"{indent}if clamp_limit is None:  # {_MARK}\n"
        f"{indent}    clamp_limit, alpha, beta = "
        f"{SWIGLU_LIMIT}, {SWIGLU_ALPHA}, {SWIGLU_BETA}"
    )
    return text[: m.start()] + replacement + text[m.end() :], True, True


def _patch_fused_ar_cudagraph(text: str) -> tuple[str, bool, bool]:
    """Skip FlashInfer fused AR when CUDA graphs are on; use NCCL fallback."""
    if _CG_AR_MARK in text:
        return text, False, True

    anchor = (
        'def _can_use_flashinfer(hidden_states: torch.Tensor, tp_size: int) -> tuple[bool, int]:\n'
        '    """Whether the flashinfer fused path applies; returns (ok, max_token_num)."""'
    )
    if anchor not in text:
        return text, False, False

    injection = (
        "    # llmc M3 cudagraph: FlashInfer fused AR+RMSNorm is not capturable on TP8\n"
        "    # (illegal memory access at capture_end; vLLM #46253). Use NCCL fallback.\n"
        f"    # {_CG_AR_MARK}\n"
        "    try:\n"
        "        from vllm.config import get_current_vllm_config\n"
        "\n"
        "        vc = get_current_vllm_config()\n"
        "        if vc is not None and not vc.enforce_eager:\n"
        "            return False, 0\n"
        "    except Exception:\n"
        "        pass\n"
    )
    new_text = text.replace(anchor, anchor + "\n" + injection, 1)
    return new_text, True, True


def _patch_moe_router_cudagraph(text: str) -> tuple[str, bool, bool]:
    """Sanitize NaN router logits at the real MoE routing entry (cudagraph padding).

    Injects ``router_logits = torch.nan_to_num(...)`` in ``RouterBase._select_experts``
    (``fused_moe/router/base_router.py``) — the template method every router
    subclass funnels through — right before it delegates to ``_compute_routing``.
    This is where vLLM maintainers pointed for the #39288 class of IMA (padding
    tokens → NaN/garbage logits → duplicate/OOB expert IDs → CUTLASS MoE out-of-
    bounds during graph capture). One edit covers fused_topk / grouped_topk / bias
    / custom routers.

    The anchor is the ``topk_weights, topk_ids = self._compute_routing(`` call,
    which appears once. Insert the sanitizer on the line before it, at the same
    indentation.
    """
    if _CG_MOE_MARK in text:
        return text, False, True

    pattern = re.compile(
        r"(?P<ind>[ \t]+)topk_weights, topk_ids = self\._compute_routing\(",
    )
    m = pattern.search(text)
    if m is None:
        return text, False, False

    ind = m.group("ind")
    injection = (
        f"{ind}# llmc M3 cudagraph: padding tokens -> NaN/garbage router logits ->\n"
        f"{ind}# duplicate/OOB expert IDs -> W4A8 CUTLASS MoE illegal memory access\n"
        f"{ind}# during graph capture (vLLM #39288 / #39391). No-op on real logits.\n"
        f"{ind}# {_CG_MOE_MARK}\n"
        f"{ind}router_logits = torch.nan_to_num(\n"
        f"{ind}    router_logits, nan=0.0, posinf=0.0, neginf=0.0\n"
        f"{ind})\n"
    )
    new_text = text[: m.start()] + injection + text[m.start() :]
    return new_text, True, True


def _find_m3_moe_model_files(vllm_dir: Path) -> list[Path]:
    """Locate the vLLM module(s) that define ``class MiniMaxM3MoE``.

    The module path differs across builds (``vllm/models/minimax_m3/nvidia/model.py``
    on some, ``vllm/model_executor/models/minimax_m3*.py`` on others), so discover
    it by content instead of hard-coding.
    """
    needle = "class MiniMaxM3MoE"
    hits: list[Path] = []
    for p in vllm_dir.rglob("*.py"):
        try:
            if needle in p.read_text(encoding="utf-8"):
                hits.append(p)
        except Exception:
            continue
    return hits


_PROBE_START = f'# === {_PROBE_MARK} ('
_PROBE_END = f'# === end {_PROBE_MARK} ==='
_LOAD_AUDIT_START = f"# === {_LOAD_AUDIT_MARK} ("
_LOAD_AUDIT_END = f"# === end {_LOAD_AUDIT_MARK} ==="


def _patch_append_probe(text: str) -> tuple[str, bool, bool]:
    """(Re)inject the env-gated MoE quality probe into the M3 model module.

    Appended at end-of-module (after the class defs), so we do not depend on any
    internal code layout — only that ``MiniMaxM3MoE`` is defined in this module's
    globals, which it is by construction (this file was selected because it
    contains ``class MiniMaxM3MoE``). If a previous block exists (between the
    start/end sentinels), it is replaced in place so probe updates redeploy with
    a plain ``--probe`` rerun.
    """
    if "class MiniMaxM3MoE" not in text:
        return text, False, False

    start = text.find(_PROBE_START)
    if start != -1:
        end = text.find(_PROBE_END, start)
        if end != -1:
            end += len(_PROBE_END)
            existing = text[start:end]
            new_block = _PROBE_BLOCK.strip("\n")
            if existing.strip() == new_block.strip():
                return text, False, True  # up to date
            # Replace old block (and trim any trailing blank lines it left).
            new_text = text[:start].rstrip("\n") + "\n\n" + new_block + "\n" + text[end:].lstrip("\n")
            return new_text, True, True

    new_text = text.rstrip("\n") + "\n" + _PROBE_BLOCK
    return new_text, True, True


def _patch_append_load_audit(text: str) -> tuple[str, bool, bool]:
    """(Re)inject the env-gated routed-expert loader audit into an M3 module."""
    if "class MiniMaxM3MoE" not in text:
        return text, False, False

    start = text.find(_LOAD_AUDIT_START)
    if start != -1:
        end = text.find(_LOAD_AUDIT_END, start)
        if end != -1:
            end += len(_LOAD_AUDIT_END)
            existing = text[start:end]
            new_block = _LOAD_AUDIT_BLOCK.strip("\n")
            if existing.strip() == new_block.strip():
                return text, False, True
            new_text = (
                text[:start].rstrip("\n")
                + "\n\n"
                + new_block
                + "\n"
                + text[end:].lstrip("\n")
            )
            return new_text, True, True

    new_text = text.rstrip("\n") + "\n" + _LOAD_AUDIT_BLOCK
    return new_text, True, True


def ensure_m3_moe_probe(*, apply: bool = True) -> str:
    """Inject (idempotently) the env-gated MoE quality probe into site-packages.

    Best-effort and separate from ``ensure_vllm_m3_patches`` (the required serve
    patches): a missing/relocated M3 model file must never block serve. Returns a
    short human-readable status string. The probe is dormant unless the worker
    env has ``M3_MOE_PROBE=1``.
    """
    vllm_dir = _vllm_dir()
    files = _find_m3_moe_model_files(vllm_dir)
    if not files:
        return "skipped (no 'class MiniMaxM3MoE' found; build layout differs)"
    statuses: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        new_text, changed, found = _patch_append_probe(text)
        if not found:
            statuses.append(f"{path.name}: no MoE class")
            continue
        if changed and apply:
            path.write_text(new_text, encoding="utf-8")
            statuses.append(f"{path.name}: injected")
        elif changed and not apply:
            statuses.append(f"{path.name}: NOT injected")
        else:
            statuses.append(f"{path.name}: already injected")
    return "; ".join(statuses)


def ensure_m3_load_audit(*, apply: bool = True) -> str:
    """Inject the dormant M3 routed-expert loader audit into site-packages."""
    vllm_dir = _vllm_dir()
    files = _find_m3_moe_model_files(vllm_dir)
    if not files:
        return "skipped (no 'class MiniMaxM3MoE' found; build layout differs)"
    statuses: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        new_text, changed, found = _patch_append_load_audit(text)
        if not found:
            statuses.append(f"{path.name}: no MoE class")
            continue
        if changed and apply:
            path.write_text(new_text, encoding="utf-8")
            statuses.append(f"{path.name}: injected")
        elif changed and not apply:
            statuses.append(f"{path.name}: NOT injected")
        else:
            statuses.append(f"{path.name}: already injected")
    return "; ".join(statuses)


def ensure_m3_quality_diagnostics(*, apply: bool = True) -> str:
    """Install all dormant MiniMax-M3 quality diagnostics."""

    probe = ensure_m3_moe_probe(apply=apply)
    audit = ensure_m3_load_audit(apply=apply)
    return f"moe_probe=[{probe}]; load_audit_and_fingerprint=[{audit}]"


def _apply(path: Path, patch_fn, check_only: bool, *, fatal: bool = True) -> bool:
    """Return True if the file is patched (already or newly)."""
    text = path.read_text(encoding="utf-8")
    new_text, changed, found = patch_fn(text)
    if not found:
        msg = f"ERROR: expected code not found in {path} (vLLM layout changed?)"
        if fatal:
            print(msg)
            sys.exit(2)
        print(msg)
        return False
    if changed and not check_only:
        path.write_text(new_text, encoding="utf-8")
        print(f"patched: {path}")
    elif changed and check_only:
        print(f"UNPATCHED: {path}")
    else:
        print(f"already patched: {path}")
    return not changed


def _patch_targets(vllm_dir: Path) -> list[tuple[str, Path, object]]:
    return [
        ("W4A8 SWIGLU support", vllm_dir / "model_executor/layers/fused_moe/experts/cutlass_moe.py", _patch_supports_activation),
        ("W4A8 SWIGLU clamp", vllm_dir / "model_executor/layers/fused_moe/activation.py", _patch_apply_activation),
        ("cudagraph fused AR", vllm_dir / "model_executor/layers/fused_allreduce_gemma_rms_norm.py", _patch_fused_ar_cudagraph),
        ("cudagraph MoE router", vllm_dir / "model_executor/layers/fused_moe/router/base_router.py", _patch_moe_router_cudagraph),
    ]


def ensure_vllm_m3_patches(*, apply: bool = True) -> None:
    """Apply (if needed) and verify all four persistent vLLM M3 serve patches.

  vLLM worker subprocesses are spawned fresh — in-process monkeypatches in
  ``serve_verify`` do **not** reach ``Worker_TP*``. This must edit site-packages.

  Raises RuntimeError if any patch cannot be applied or verified.
    """
    vllm_dir = _vllm_dir()
    missing_files: list[str] = []
    unpatched: list[str] = []
    for label, path, patch_fn in _patch_targets(vllm_dir):
        if not path.exists():
            missing_files.append(str(path))
            continue
        ok = _apply(path, patch_fn, check_only=not apply, fatal=False)
        if not ok:
            unpatched.append(label)

    if missing_files:
        raise RuntimeError(
            "vLLM M3 serve files not found (wrong vLLM build?). Missing:\n  "
            + "\n  ".join(missing_files)
            + "\nInstall: bash pipeline/slurm/install_vllm_m3_serve.sh"
        )
    if unpatched:
        raise RuntimeError(
            "vLLM M3 serve patches missing in site-packages: "
            + ", ".join(unpatched)
            + ". Run: python pipeline/slurm/patch_vllm_m3_serve.py"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only; exit 1 if unpatched")
    ap.add_argument(
        "--probe",
        action="store_true",
        help="also inject the env-gated MoE quality probe (M3_MOE_PROBE=1 at serve)",
    )
    args = ap.parse_args()

    vllm_dir = _vllm_dir()
    for label, path, _ in _patch_targets(vllm_dir):
        if not path.exists():
            print(f"ERROR: {path} not found ({label}); is this the W4A8-MoE vLLM build?")
            return 2

    import vllm

    print(f"vLLM {getattr(vllm, '__version__', '?')} at {vllm_dir}")
    _report_flashinfer_version()
    results = [
        _apply(path, patch_fn, args.check)
        for _, path, patch_fn in _patch_targets(vllm_dir)
    ]

    if args.check:
        probe_status = ensure_m3_moe_probe(apply=False)
        print(f"MoE quality probe: {probe_status}")
        already = all(results)
        print("STATUS:", "patched" if already else "NOT patched")
        return 0 if already else 1

    if args.probe:
        print(f"MoE quality probe: {ensure_m3_moe_probe(apply=True)}")
        print(
            "  Enable at serve time with: M3_MOE_PROBE=1 (optional "
            "M3_MOE_PROBE_RECOMPUTE=1 to also log routed-only norm)."
        )

    print(
        "\nDone. Recompile of C++/CUDA is NOT required (pure-Python edits).\n"
        "Re-run after any vLLM reinstall. Then serve normally, e.g.:\n"
        "  vllm serve <ckpt> --tensor-parallel-size 8 --enable-expert-parallel ..."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
