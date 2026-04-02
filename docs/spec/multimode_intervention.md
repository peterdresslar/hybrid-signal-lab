# Gain Intervention Mode: Implementation Specification

**Issue context:** The existing codebase applies gain scaling by hooking the full
decoder layer (block) output, which scales `g * (residual + attention + FF)`. The
manuscript formalizes gain as scaling only the attention contribution to the
residual stream: `g * ã`. This spec defines a principled, architecture-aware
implementation that aligns the code with the formalism.

---

## 1. Algebraic foundation

### 1.1 Unified gain equation

For any hybrid model with softmax attention layers at positions $l \in S$, the
gain intervention modifies the residual stream update:

$$h_i^l = h_i^{l-1} + g^l \cdot \tilde{a}_i^l + \tilde{f}_i^l$$

where:

| Symbol | Definition |
|---|---|
| $h_i^l$ | Hidden state of token $i$ after layer $l$ (residual stream) |
| $h_i^{l-1}$ | Hidden state of token $i$ entering layer $l$ (residual input) |
| $g^l$ | Gain scalar for layer $l$ (from profile; $g^l = 1$ at non-softmax layers) |
| $\tilde{a}_i^l$ | Attention contribution at the point it enters the residual stream |
| $\tilde{f}_i^l$ | Feed-forward contribution at the point it enters the residual stream |

The tilde notation $\tilde{a}$ is critical: it denotes the attention signal **as
the residual stream receives it**, which differs by architecture due to
normalization placement.

### 1.2 Qwen 3.5 Hybrid: pre-norm

Source: `transformers/models/qwen3_5/modeling_qwen3_5.py`, class
`Qwen3_5DecoderLayer`. Single class handles both layer types via
`self.layer_type` branch.

Forward (softmax layers, `layer_type == "full_attention"`):
```python
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)       # N_in
hidden_states, _ = self.self_attn(hidden_states, ...)     # Attn
hidden_states = residual + hidden_states                  # residual add
# ... then post_attention_layernorm → mlp → residual add for FF
```

Algebra:
$$h_i^l = h_i^{l-1} + \underbrace{\mathrm{Attn}^l(\mathrm{N_{in}}(h_i^{l-1}))}_{\tilde{a}_i^l} + \tilde{f}_i^l$$

The norm is **before** attention (`input_layernorm`). The raw attention output
goes directly to the residual add. Therefore:

$$\tilde{a}_i^l = \mathrm{Attn}^l(\mathrm{N_{in}}(h_i^{l-1}))$$

**Hook target for `attention_contribution` mode:** `self_attn`
- Scaling this module's output by $g$ gives $g \cdot \mathrm{Attn}^l(\mathrm{N_{in}}(h_i^{l-1}))$
- This propagates directly into the residual add ✓

### 1.3 OLMo Hybrid: post-norm (softmax layers only)

Source: `transformers/models/olmo_hybrid/modeling_olmo_hybrid.py`, class
`OlmoHybridAttentionDecoderLayer`. Separate class from
`OlmoHybridLinearAttentionDecoderLayer`.

Forward (softmax / full-attention layers):
```python
residual = hidden_states
hidden_states, _ = self.self_attn(hidden_states, ...)              # Attn (no input norm!)
hidden_states = self.post_attention_layernorm(hidden_states)       # N_att
hidden_states = residual + hidden_states                           # residual add
# ... then mlp → post_feedforward_layernorm → residual add for FF
```

Algebra:
$$h_i^l = h_i^{l-1} + \underbrace{\mathrm{N_{att}}(\mathrm{Attn}^l(h_i^{l-1}))}_{\tilde{a}_i^l} + \tilde{f}_i^l$$

The norm is **after** attention (`post_attention_layernorm`), sitting between the
attention output and the residual add. Therefore:

$$\tilde{a}_i^l = \mathrm{N_{att}}(\mathrm{Attn}^l(h_i^{l-1}))$$

**Why hooking `self_attn` does not work:** RMSNorm is scale-invariant
($\mathrm{RMSNorm}(g \cdot x) = \mathrm{RMSNorm}(x)$). Scaling before the norm
has zero effect on the residual stream.

**Hook target for `attention_contribution` mode:** `post_attention_layernorm`
- Scaling this module's output by $g$ gives $g \cdot \mathrm{N_{att}}(\mathrm{Attn}^l(h_i^{l-1}))$
- This is the value that enters the residual add ✓

### 1.4 OLMo Hybrid: pre-norm (linear/GDN layers)

Note for completeness — the GDN layers in OLMo Hybrid use pre-norm:
```python
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)
hidden_states = self.linear_attn(hidden_states, ...)
hidden_states = residual + hidden_states
```

This means OLMo Hybrid is internally asymmetric: pre-norm for GDN layers,
post-norm for softmax layers. Since we only apply gain at softmax layers, we only
encounter the post-norm case.

### 1.5 Equivalence

Despite different hook targets, the intervention has **identical algebraic
semantics** on both architectures:

$$h_i^l = h_i^{l-1} + g^l \cdot \tilde{a}_i^l + \tilde{f}_i^l$$

The hook location differs to account for normalization placement, but what gets
scaled — the attention contribution as the residual stream receives it — is the
same.

### 1.6 Block-output mode (existing behavior)

The existing codebase hooks the full decoder layer. The decoder layer's output is:

$$\mathrm{Layer}^l(h_i^{l-1}) = h_i^{l-1} + \tilde{a}_i^l + \tilde{f}_i^l = h_i^l$$

So block-output scaling gives:

$$g^l \cdot (h_i^{l-1} + \tilde{a}_i^l + \tilde{f}_i^l)$$

which is **not** the same as the gain equation — it scales the residual
pass-through and the FF contribution as well. Battery 4 results used this mode.
This is a valid intervention (it changes the model's behavior), but it is a
different one than Equation 2 describes. Both modes should be available.

---

## 2. Intervention modes

Three named modes, as a string enum:

| Mode | Semantics | Use case |
|---|---|---|
| `block_output` | Hook full decoder layer; scale entire output including residual pass-through and FF | Reproduces Battery 4 results exactly |
| `attention_contribution` | Hook the attention contribution at its residual-entry point (architecture-aware) | Matches the manuscript's gain equation |
| `backend_default` | Resolves to whichever mode the backend declares as its legacy default | Convenience for backward compatibility |

### 2.1 Default resolution

| Backend | `backend_default` resolves to | Rationale |
|---|---|---|
| `QwenBackend` | `block_output` | Battery 4 hooked full layer |
| `OlmoBackend` | `block_output` | Battery 4 hooked full layer |
| `TransformerBackend` | `attention_contribution` | Legacy code hooked `self_attn` directly |

---

## 3. Hook target table

| Backend | `block_output` target | `attention_contribution` target |
|---|---|---|
| `QwenBackend` | `model.model.layers[idx]` | `model.model.layers[idx].self_attn` |
| `OlmoBackend` | `model.model.layers[idx]` | `model.model.layers[idx].post_attention_layernorm` |
| `TransformerBackend` | `model.model.layers[idx]` (or equivalent via `get_decoder_layers`) | Attention submodule via `get_layer_attention_module()` |

### 3.1 Qwen structural note

Qwen uses one `Qwen3_5DecoderLayer` class for all layers, branching on
`self.layer_type`. Softmax layers have `self.self_attn`; GDN layers have
`self.linear_attn` and no `self_attn`. Since `get_attention_layer_indices()`
already filters to softmax-only layers, hooking `self_attn` on those indices is
safe.

### 3.2 OLMo structural note

OLMo uses two separate classes: `OlmoHybridAttentionDecoderLayer` (softmax,
has `self_attn` + `post_attention_layernorm`) and
`OlmoHybridLinearAttentionDecoderLayer` (GDN, has `linear_attn` +
`input_layernorm`). The attention layers identified by
`get_attention_layer_indices()` will all be instances of
`OlmoHybridAttentionDecoderLayer`, so `post_attention_layernorm` is guaranteed to
exist on them.

---

## 4. Implementation checklist

All changes are additive — no existing behavior changes unless a new flag is
passed.

### 4.1 `backend.py`

- [ ] Add `InterventionMode` string enum: `"block_output"`, `"attention_contribution"`, `"backend_default"`
- [ ] Update `get_hook_module()` signature to accept `mode: InterventionMode`
- [ ] Add `default_intervention_mode` property to `ModelBackend` ABC (returns `InterventionMode.BLOCK_OUTPUT`)
- [ ] Add `resolve_intervention_mode()` method: if `backend_default`, return `self.default_intervention_mode`; otherwise return the mode as-is
- [ ] Docstring on `InterventionMode` should contain the unified gain equation and explain that `ã` differs by architecture

### 4.2 `qwen.py`

- [ ] Override `default_intervention_mode` → `InterventionMode.BLOCK_OUTPUT`
- [ ] Update `get_hook_module(layer_idx, mode)`:
  - `BLOCK_OUTPUT`: return `self.model.model.layers[layer_idx]`
  - `ATTENTION_CONTRIBUTION`: return `self.model.model.layers[layer_idx].self_attn`
- [ ] Docstring: "Pre-norm architecture. Attn output enters residual directly, so hooking self_attn scales ã = Attn(N_in(x))."

### 4.3 `olmo.py`

- [ ] Override `default_intervention_mode` → `InterventionMode.BLOCK_OUTPUT`
- [ ] Update `get_hook_module(layer_idx, mode)`:
  - `BLOCK_OUTPUT`: return `self.model.model.layers[layer_idx]`
  - `ATTENTION_CONTRIBUTION`: return `self.model.model.layers[layer_idx].post_attention_layernorm`
- [ ] Docstring: "Post-norm architecture on softmax layers. RMSNorm(g·x) = RMSNorm(x), so we must hook post_attention_layernorm to scale ã = N_att(Attn(x))."

### 4.4 `transformer.py`

- [ ] Override `default_intervention_mode` → `InterventionMode.ATTENTION_CONTRIBUTION`
- [ ] Update `get_hook_module(layer_idx, mode)`:
  - `BLOCK_OUTPUT`: return full decoder layer
  - `ATTENTION_CONTRIBUTION`: return attention submodule (existing behavior)
- [ ] Docstring: note this is the generic fallback; architecture-specific backends should be preferred

### 4.5 `agent.py`

- [ ] Add `intervention_mode` parameter to `run_pass()` (default `"backend_default"`)
- [ ] Call `self.backend.resolve_intervention_mode(mode)` to get the concrete mode
- [ ] Pass resolved mode to `get_hook_module()` in the hook registration loop
- [ ] Same for `score_target()` if it registers hooks
- [ ] Include resolved mode string in the result dict

### 4.6 `signal_lab.py` (CLI)

- [ ] Add `--gain-mode` argument: choices `["backend_default", "block_output", "attention_contribution"]`, default `"backend_default"`
- [ ] Pass through to `agent.run_pass()`

### 4.7 `sweep.py`

- [ ] Add `gain_intervention_mode` to sweep config / parameter grid
- [ ] Pass through to `agent.run_pass()`
- [ ] Record in sweep results metadata

### 4.8 Validation

- [ ] Unit: for each backend, assert `get_hook_module(idx, BLOCK_OUTPUT)` returns the decoder layer and `get_hook_module(idx, ATTENTION_CONTRIBUTION)` returns the correct submodule
- [ ] Smoke: one Qwen prompt + one OLMo prompt, compare logits under `block_output` vs `attention_contribution` at the same gain profile — they should differ (if they don't, something is wrong)
- [ ] Regression: run a known Battery 4 prompt under `block_output` mode and confirm the result matches the stored baseline

---

## 5. What NOT to change

- The `attention_scaler_hook()` function itself is fine — it multiplies module
  output by a scalar, and that works regardless of what module it's attached to.
- The gain profile system (`g_profile.py`) is unchanged — profiles still produce
  a vector of scalars indexed by attention slot.
- Battery 4 results remain valid — they used `block_output` semantics, and that
  mode is preserved as the default.

---

## 6. Manuscript implications

Once `attention_contribution` mode is implemented and validated:

1. **Equation 2** in the manuscript becomes directly executable — the code does
   exactly what the equation says.
2. **Section 3 (Methods)** should note both modes and explain that Battery 4 used
   `block_output`; future batteries can use `attention_contribution`.
3. The **comparison** between modes is itself an interesting result: if they
   diverge substantially, it tells us something about the relative contribution
   of the FF pathway to the intervention's observed effects.