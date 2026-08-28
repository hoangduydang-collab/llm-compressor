import contextlib
from typing import TYPE_CHECKING, Iterator

import torch
from compressed_tensors.offload import disable_offloading, set_onload_device
from loguru import logger
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from llmcompressor.core import LifecycleCallbacks, active_session
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.registry import CalibrationPipeline
from llmcompressor.pipelines.sequential.helpers import (
    handle_sequential_oom,
    trace_subgraphs,
)
from llmcompressor.utils.dev import get_main_device
from llmcompressor.utils.helpers import DisableQuantization, calibration_forward_context
from llmcompressor.utils.pytorch.module import infer_sequential_targets

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments

__all__ = ["SequentialPipeline"]


def _invoke_sequential_trace_callback(state, diagnostics: dict) -> None:
    """Invoke an opt-in guard immediately after the production FX trace."""
    callback = getattr(state, "sequential_trace_callback", None)
    if callback is not None:
        callback(diagnostics)


def _invoke_post_sequential_propagation_callback(
    state,
    *,
    subgraph_index: int,
    num_subgraphs: int,
    modules: list[torch.nn.Module],
    propagated: bool,
) -> None:
    """Invoke an opt-in diagnostic boundary after sequential propagation."""
    callback = getattr(state, "post_sequential_propagation_callback", None)
    if callback is not None:
        callback(
            subgraph_index=subgraph_index,
            num_subgraphs=num_subgraphs,
            modules=modules,
            propagated=propagated,
        )


def _targeted_module_names(model: torch.nn.Module) -> set[str]:
    """Names of modules a modifier has scheduled for quantization.

    ``quantization_scheme`` is attached by ``apply_quantization_config`` during
    modifier initialization, which runs before the pipeline, so this is already
    populated by the time subgraphs are traced.
    """
    return {
        name
        for name, module in model.named_modules()
        if getattr(module, "quantization_scheme", None) is not None
    }


def _subgraph_module_prefixes(subgraph) -> set[str]:
    """Module paths this subgraph calls directly, as name prefixes.

    Deliberately reads the graph's ``call_module`` targets rather than
    ``subgraph.submodules()``: on models where an untraceable call is wrapped by
    the AST autowrapper (e.g. ``self.experts(...).view(*orig_shape)`` -- fx
    cannot trace ``*args`` unpacking, so the whole expression is wrapped), the
    wrapped modules have no node of their own. Prefix matching still attributes
    them to the subgraph via an ancestor node, and the caller verifies coverage
    rather than trusting this.
    """
    return {node.target for node in subgraph.graph.find_nodes(op="call_module")}


def last_subgraph_with_targets(model: torch.nn.Module, subgraphs: list) -> int | None:
    """Index of the last subgraph holding a module scheduled for compression.

    Returns ``None`` when the answer cannot be established safely, in which case
    the caller must walk every subgraph. That is the fail-closed direction: a
    false negative here would stop the pipeline before a layer that should have
    been quantized, silently shipping it unquantized.

    The safety condition is coverage: every targeted module must be attributable
    to some subgraph at or before the returned index. If even one targeted module
    cannot be placed, we refuse to skip anything.
    """
    targeted = _targeted_module_names(model)
    if not targeted:
        return None

    covered: set[str] = set()
    last: int | None = None
    for index, subgraph in enumerate(subgraphs):
        prefixes = _subgraph_module_prefixes(subgraph)
        hits = {
            name
            for name in targeted
            if any(name == p or name.startswith(f"{p}.") for p in prefixes)
        }
        if hits:
            covered |= hits
            last = index

    if last is None or covered != targeted:
        return None
    return last


def _get_batches(
    activations: IntermediatesCache,
    num_batches: int,
    input_names: list[str],
    desc: str,
    sequential_prefetch: bool = False,
) -> Iterator[tuple[int, dict]]:
    """
    Yield (batch_idx, inputs) with the next batch optionally prefetched in a
    background thread to overlap fetch (onload from offload device) with the
    main-thread forward pass. Delegates to
    :meth:`IntermediatesCache.iter_prefetch` when prefetching is enabled.
    """
    batch_source = (
        activations.iter_prefetch(input_names)
        if sequential_prefetch
        else activations.iter(input_names)
    )
    for batch_idx, inputs in tqdm(
        enumerate(batch_source), total=num_batches, desc=desc
    ):
        yield batch_idx, inputs


@CalibrationPipeline.register("sequential")
class SequentialPipeline(CalibrationPipeline):
    @staticmethod
    @handle_sequential_oom
    def __call__(
        model: torch.nn.Module,
        dataloader: DataLoader,
        dataset_args: "DatasetArguments",
    ):
        """
        Run a sequential data pipeline according to the following steps:

        1. The model is partitioned into subgraphs according to `sequential_targets`
        2. Data passes through each subgraph sequentially. Data is passed through each
            subgraph twice, once to trigger calibration hooks, then a second time in
            order to capture activations after quantization has occurred through hooks.
        3. The intermediate activations between each subgraph are cached and offloaded
            to the cpu between each batch in order to save memory

        This pipeline requires that the model be traceable with respect to data from the
        data loader. This may be an issue for vision models with vision datasets, due
        to specialized input processing in the model.

        In the event that tracing fails, a torch.fx.proxy.TraceError will be raised. A
        model can be made traceable by wrapping the untraceable functions (see
        llmcompressor.transformers.tracing)

        :param model: model being calibrated
        :param dataloader: loads data for calibration
        :param dataset_args: dataset arguments relevant to pipelines
        """
        session = active_session()

        # prepare model for sequential onloading
        onload_device = get_main_device()
        offload_device = torch.device(dataset_args.sequential_offload_device)
        set_onload_device(model, onload_device)

        # AutoRoundModifier optimizes each layer independently using its own
        # forward passes, so quantization error should not be propagated between
        # layers during the calibration stage
        modifiers = session.lifecycle.recipe.modifiers
        if any(type(m).__name__ == "AutoRoundModifier" for m in modifiers):
            dataset_args.propagate_error = False

        # prepare to trace subgraphs
        sequential_targets = infer_sequential_targets(
            model, dataset_args.sequential_targets
        )
        ignore = dataset_args.tracing_ignore

        # trace subgraphs
        sample_input = next(iter(dataloader))
        trace_diagnostics = (
            {} if getattr(session.state, "sequential_trace_callback", None) else None
        )
        subgraphs = trace_subgraphs(
            model,
            sample_input,
            sequential_targets,
            ignore,
            dataset_args.sequential_targets_per_subgraph,
            diagnostics=trace_diagnostics,
        )
        if trace_diagnostics is not None:
            _invoke_sequential_trace_callback(session.state, trace_diagnostics)
        num_subgraphs = len(subgraphs)

        LifecycleCallbacks.calibration_start()

        with contextlib.ExitStack() as stack:
            stack.enter_context(calibration_forward_context(model))
            stack.enter_context(DisableQuantization(model))
            # prepare intermediates cache
            activations = IntermediatesCache.from_dataloader(
                dataloader, onload_device, offload_device
            )

            # Populate loss_masks once from cached activations for AWQ masking support
            use_loss_mask = getattr(dataset_args, "use_loss_mask", False)
            if use_loss_mask:
                session.state.loss_masks = [
                    activations.fetch(batch_idx, ["loss_mask"]).get("loss_mask")
                    for batch_idx in range(len(dataloader))
                ]
            else:
                session.state.loss_masks = None

            sequential_prefetch = getattr(dataset_args, "sequential_prefetch", False)
            session.state.sequential_prefetch = sequential_prefetch

            # SMOKE ONLY. Trailing subgraphs with nothing to compress exist only
            # to produce inputs for subgraphs after them, so once nothing after
            # them is targeted, walking them is pure cost -- measured at ~2.5 min
            # per layer on GLM-5.2, which is 19 GB of weights at network-storage
            # speed. last_subgraph_with_targets returns None unless it can
            # account for EVERY targeted module at or before the index, so an
            # unattributable target walks the whole model rather than risking a
            # layer being skipped unquantized.
            stop_at = None
            if getattr(dataset_args, "stop_after_last_target", False):
                stop_at = last_subgraph_with_targets(model, subgraphs)
                if stop_at is None:
                    logger.warning(
                        "stop_after_last_target=True but the last targeted "
                        "subgraph could not be established safely (no module "
                        "carries a quantization_scheme, or a targeted module "
                        "could not be attributed to any subgraph). Walking all "
                        f"{num_subgraphs} subgraphs."
                    )
                elif stop_at < num_subgraphs - 1:
                    logger.warning(
                        f"stop_after_last_target=True: stopping after subgraph "
                        f"{stop_at + 1}/{num_subgraphs}, skipping "
                        f"{num_subgraphs - stop_at - 1} trailing subgraphs with "
                        "nothing to compress. Those layers remain unquantized, "
                        "which is what the ignore list already asked for -- but "
                        "any modifier needing statistics over the whole model "
                        "will see a truncated model. Do not use in production."
                    )

            for subgraph_index, subgraph in enumerate(subgraphs):
                # prepare tqdm description texts
                calib_desc = f"({subgraph_index + 1}/{num_subgraphs}): Calibrating"
                prop_desc = f"({subgraph_index + 1}/{num_subgraphs}): Propagating"

                # reduce memory movement by keeping modules onloaded
                num_batches = len(dataloader)
                with disable_offloading():
                    subgraph_modules = subgraph.submodules(model)
                    # do a preliminary pass to trigger modifier hooks
                    for batch_idx, inputs in _get_batches(
                        activations,
                        num_batches,
                        subgraph.input_names,
                        calib_desc,
                        sequential_prefetch,
                    ):
                        session.state.current_batch_idx = batch_idx
                        outputs = subgraph.forward(model, **inputs)

                        if not dataset_args.propagate_error:
                            if subgraph_index < num_subgraphs - 1:
                                activations.update(batch_idx, outputs)
                                activations.delete(batch_idx, subgraph.consumed_names)

                    LifecycleCallbacks.sequential_epoch_end(subgraph_modules)

                    if dataset_args.propagate_error:
                        # this pass does not trigger modifier hooks
                        # and is only used for capturing outputs of compressed modules
                        with HooksMixin.disable_hooks():
                            for batch_idx, inputs in _get_batches(
                                activations,
                                num_batches,
                                subgraph.input_names,
                                prop_desc,
                                sequential_prefetch,
                            ):
                                output = subgraph.forward(model, **inputs)
                                if subgraph_index < num_subgraphs - 1:
                                    activations.update(batch_idx, output)
                                    activations.delete(
                                        batch_idx, subgraph.consumed_names
                                    )

                    _invoke_post_sequential_propagation_callback(
                        session.state,
                        subgraph_index=subgraph_index,
                        num_subgraphs=num_subgraphs,
                        modules=subgraph_modules,
                        propagated=dataset_args.propagate_error,
                    )

                # Outside `disable_offloading()` so this subgraph is offloaded
                # before we leave, exactly as a normal iteration would.
                if stop_at is not None and subgraph_index >= stop_at:
                    break

            # redundant, finish any remaining compression
            LifecycleCallbacks.calibration_end()
