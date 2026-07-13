# MiniMax-M3 MoE Forward and Linearization Analysis

Date: 2026-07-13

## Purpose

This report supplies the planner with the remote MiniMax-M3 MoE implementation
and the runtime type observed after applying `linearize_moe`. It investigates
why all 129 resolved AWQ expert mappings had zero forward-hook events during
calibration.

## Source inspected

The active quant environment contains the canonical MiniMax implementation at:

```text
/mnt/nfs/hoangduy/venvs/quant/lib/python3.12/site-packages/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py
```

The relevant classes are `MiniMaxM3VLExperts` and
`MiniMaxM3VLSparseMoeBlock`.

## Original MiniMax expert implementation

`MiniMaxM3VLExperts` stores all routed expert weights in stacked tensors:

```python
class MiniMaxM3VLExperts(nn.Module):
    def __init__(self, config):
        ...
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                2 * self.intermediate_dim,
                self.hidden_dim,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.hidden_dim,
                self.intermediate_dim,
            )
        )

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(
                top_k_index, num_classes=self.num_experts
            ).permute(2, 1, 0)
            hit = torch.greater(
                mask.sum(dim=(-1, -2)), 0
            ).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            current = self._apply_gate(
                F.linear(
                    hidden_states[token_idx],
                    self.gate_up_proj[expert_idx],
                )
            )
            current = F.linear(
                current,
                self.down_proj[expert_idx],
            ) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(
                0, token_idx, current.to(final.dtype)
            )
        return final
```

Therefore, the original remote implementation does **not** call individual
expert `nn.Linear` modules. It performs `F.linear` directly against slices of
the stacked expert tensors.

## Sparse-MoE block dispatch

The sparse block itself calls the experts object as a module:

```python
class MiniMaxM3VLSparseMoeBlock(nn.Module):
    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        shared_output = self.shared_experts(hidden_states)

        _, routing_weights, selected_experts = self.gate(hidden_states)
        hidden_states = self.experts(
            hidden_states,
            selected_experts,
            routing_weights,
        )
        hidden_states = hidden_states * self.routed_scaling_factor
        hidden_states = hidden_states + shared_output

        return hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
```

Thus, before linearization, `self.experts(...)` enters the fused
`MiniMaxM3VLExperts.forward`, which bypasses per-expert module forwards.

## Post-linearization type check

I built the meta model using the exact production full-calibration config:

```text
pipeline/configs/minimax_m3_full_calib.yaml
```

After `linearize_moe`, the layer-8 object is:

```text
model.model.language_model.layers[8].mlp.experts
  type: llmcompressor.modeling.moe.linear_experts.LinearExperts2D
  module: llmcompressor.modeling.moe.linear_experts
  MRO:
    LinearExperts2D
    LinearExperts2D
    torch.nn.modules.container.ModuleList
    torch.nn.modules.module.Module
  experts[0]:
    llmcompressor.modeling.moe.linear_experts.ExpertMLPWithGate
```

The duplicate `LinearExperts2D` entry in the MRO is the dynamically generated
target-compatible subclass; it still derives from the repository's
`LinearExperts2D` implementation.

## Important implication

The original MiniMax implementation explains why the unmodified model bypasses
per-expert hooks. However, it is not by itself a complete explanation for the
observed calibration result after `linearize_moe`.

The repository replacement's forward implementation loops over expert indices
and invokes the child module:

```python
for expert_index in range(self.num_experts):
    ...
    expert = self[expert_index]
    expert_output = expert(hidden_states[token_indices])
```

Therefore, if the calibration forward is using the linearized object above,
the `ExpertMLPWithGate` modules should execute and hooks attached to them
should fire. The observed values:

```text
resolved_mapping_count=129
total_balance_forward_events=0
balance_layers_never_fired_count=129
```

indicate that calibration is likely executing a different object/forward path
than the meta-model inspection, or that FX/sequential preparation bypasses the
linearized `self.experts` object after hooks are installed.

## Recommended next diagnostic

At calibration runtime, immediately before the first representative forward,
log all of the following for layer 8:

```python
print(type(model.model.language_model.layers[8].mlp.experts))
print(type(model.model.language_model.layers[8].mlp.experts[0]))
print(model.model.language_model.layers[8].mlp.experts[0].forward)
```

Also register one temporary hook directly on:

```python
model.model.language_model.layers[8].mlp.experts
model.model.language_model.layers[8].mlp.experts[0]
model.model.language_model.layers[8].mlp.experts[0].gate_proj
```

This distinguishes:

1. calibration retaining the linearized module but not entering the MoE block;
2. calibration replacing or cloning the original fused experts object;
3. the linearized container running while child expert forwards are bypassed;
4. hooks being attached to stale module objects after FX/sequential rewriting.

No GPU job was launched for this investigation. The type check used a CPU-only
meta model and the exact production full-calibration configuration.
