"""Signal Lab: exploring model internals via transformers.

Run with: uv run python -m colony.signal_lab
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import dotenv

# ensure HF_TOKEN loaded from .env_development
dotenv.load_dotenv(".env.development")
parser = argparse.ArgumentParser(description="Signal Lab: exploring model internals via transformers.")
parser.add_argument("--prompt", type=str, help="Path or filename to a prompt file in data/, or a direct string prompt.")
args = parser.parse_args()

# Step 1: Load model and tokenizer
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_NAME = "Qwen/Qwen3.5-2B"

DEFAULT_PROMPT = "The color with the shortest wavelength is"

if args.prompt:
    if os.path.isfile(args.prompt):
        with open(args.prompt, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
    else:
        data_path = os.path.join("data", args.prompt)
        if os.path.isfile(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        else:
            prompt = args.prompt
else:
    prompt = DEFAULT_PROMPT

print(f"Device: {DEVICE}")
print(f"Loading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map=DEVICE,
    attn_implementation="eager",
)
model.eval()

print(f"Model loaded: {model.config.num_hidden_layers} layers, "
      f"hidden_size={model.config.hidden_size}")

# Step 2: Run a single forward pass requesting everything

inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

print(f"\nPrompt: '{prompt}'")
print(f"Input token IDs: {inputs['input_ids'].tolist()}")
print(f"Input tokens decoded: {[tokenizer.decode(t) for t in inputs['input_ids'][0]]}")
print(f"Input keys: {list(inputs.keys())}")
for k, v in inputs.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True,
        output_attentions=True,
    )

# ============================================================
# FULL INVENTORY: everything in the outputs object
# ============================================================
print("\n" + "=" * 60)
print("FULL OUTPUT INVENTORY")
print("=" * 60)

# What type is it?
print(f"\nOutput type: {type(outputs).__name__}")
print(f"Output class: {type(outputs).__mro__}")

# What keys/attributes are available?
if hasattr(outputs, 'keys'):
    print(f"\nOutput keys: {list(outputs.keys())}")

# Walk every attribute
print(f"\nAll attributes on outputs object:")
for attr in dir(outputs):
    if attr.startswith('_'):
        continue
    val = getattr(outputs, attr)
    if callable(val):
        print(f"  {attr}: <method>")
    elif isinstance(val, torch.Tensor):
        print(f"  {attr}: Tensor shape={val.shape}, dtype={val.dtype}")
    elif isinstance(val, tuple):
        if len(val) > 0 and isinstance(val[0], torch.Tensor):
            print(f"  {attr}: tuple of {len(val)} Tensors, "
                  f"first shape={val[0].shape}, dtype={val[0].dtype}")
        elif len(val) > 0 and isinstance(val[0], tuple):
            # nested tuples (like past_key_values)
            inner = val[0]
            if len(inner) > 0 and isinstance(inner[0], torch.Tensor):
                print(f"  {attr}: tuple of {len(val)} tuples, "
                      f"each with {len(inner)} Tensors, "
                      f"inner shape={inner[0].shape}, dtype={inner[0].dtype}")
            else:
                print(f"  {attr}: tuple of {len(val)} tuples")
        else:
            print(f"  {attr}: tuple of {len(val)} items, "
                  f"first type={type(val[0]).__name__ if val else 'empty'}")
    elif val is None:
        print(f"  {attr}: None")
    else:
        print(f"  {attr}: {type(val).__name__} = {repr(val)[:100]}")

# ============================================================
# DETAILED BREAKDOWN of each output field
# ============================================================
print("\n" + "=" * 60)
print("DETAILED BREAKDOWN")
print("=" * 60)

# --- Logits ---
print(f"\n--- logits ---")
print(f"Shape: {outputs.logits.shape}")
print(f"  [batch={outputs.logits.shape[0]}, "
      f"seq_len={outputs.logits.shape[1]}, "
      f"vocab_size={outputs.logits.shape[2]}]")
print(f"Dtype: {outputs.logits.dtype}")
# Per-position entropy (how uncertain is the model at each token?)
for pos in range(outputs.logits.shape[1]):
    pos_logits = outputs.logits[0, pos, :].float()
    pos_probs = torch.softmax(pos_logits, dim=-1)
    entropy = -(pos_probs * torch.log2(pos_probs + 1e-10)).sum()
    top_token = tokenizer.decode(pos_logits.argmax())
    input_token = tokenizer.decode(inputs['input_ids'][0, pos])
    print(f"  Position {pos} (input='{input_token}'): "
          f"entropy={entropy:.2f} bits, top_next='{top_token}'")

# --- Hidden states ---
print(f"\n--- hidden_states ---")
print(f"Number of layers: {len(outputs.hidden_states)} "
      f"(= {len(outputs.hidden_states) - 1} transformer layers + embedding)")
for i, hs in enumerate(outputs.hidden_states):
    label = "embedding" if i == 0 else f"layer {i}"
    # stats at the last token position (the "signal" position)
    last_pos = hs[0, -1, :].float()
    print(f"  [{label}] shape={hs.shape}, dtype={hs.dtype}, "
          f"last_pos: mean={last_pos.mean():.4f}, "
          f"std={last_pos.std():.4f}, "
          f"norm={last_pos.norm():.4f}, "
          f"min={last_pos.min():.4f}, max={last_pos.max():.4f}")

# --- Attention weights ---
print(f"\n--- attentions ---")
print(f"Number of layers: {len(outputs.attentions)}")
for i, attn in enumerate(outputs.attentions):
    # attn shape: [batch, num_heads, seq_len, seq_len]
    # How much does each head attend to the first vs. last token?
    attn_to_first = attn[0, :, -1, 0].float().mean()  # last pos attending to first
    attn_to_last = attn[0, :, -1, -1].float().mean()   # last pos attending to itself
    attn_entropy = -(attn[0, :, -1, :].float() *
                     torch.log2(attn[0, :, -1, :].float() + 1e-10)).sum(dim=-1).mean()
    print(f"  Layer {i}: shape={attn.shape}, "
          f"num_heads={attn.shape[1]}, "
          f"last→first={attn_to_first:.4f}, "
          f"last→self={attn_to_last:.4f}, "
          f"attn_entropy={attn_entropy:.2f} bits")

# --- Past key values (KV cache) ---
print(f"\n--- past_key_values ---")
if outputs.past_key_values is not None:
    pkv = outputs.past_key_values
    print(f"Type: {type(pkv).__name__}")
    if hasattr(pkv, 'key_cache'):
        print(f"key_cache: {len(pkv.key_cache)} layers")
        for i, k in enumerate(pkv.key_cache):
            if k is not None:
                v = pkv.value_cache[i]
                print(f"  layer {i}: ATTENTION  key={k.shape}, value={v.shape}")
            else:
                print(f"  layer {i}: GDN        (no KV cache)")
    # Check for GDN recurrent state
    if hasattr(pkv, 'conv_states'):
        print(f"\nconv_states: {len(pkv.conv_states)} layers")
        for i, cs in enumerate(pkv.conv_states):
            if cs is not None:
                print(f"  layer {i}: shape={cs.shape}")
    if hasattr(pkv, 'recurrent_states'):
        print(f"\nrecurrent_states: {len(pkv.recurrent_states)} layers")
        for i, rs in enumerate(pkv.recurrent_states):
            if rs is not None:
                print(f"  layer {i}: shape={rs.shape}")
else:
    print("None (not returned)")

# --- Top predictions ---
print(f"\n--- Top 10 next-token predictions ---")
last_logits = outputs.logits[0, -1, :].float()
probs = torch.softmax(last_logits, dim=-1)
top10 = torch.topk(probs, 10)
total_entropy = -(probs * torch.log2(probs + 1e-10)).sum()
print(f"Total entropy at last position: {total_entropy:.2f} bits")
print(f"(max possible: {torch.log2(torch.tensor(float(probs.shape[0]))):.2f} bits)")
for rank, (prob, idx) in enumerate(zip(top10.values, top10.indices)):
    token = tokenizer.decode(idx)
    print(f"  #{rank+1} '{token}' : p={prob:.4f}, "
          f"logprob={torch.log(prob):.4f}, "
          f"token_id={idx.item()}")

# --- Model config (non-tensor metadata) ---
print(f"\n--- Model config (selected) ---")
config = model.config
for key in ['model_type', 'num_hidden_layers', 'hidden_size',
            'intermediate_size', 'num_attention_heads',
            'num_key_value_heads', 'vocab_size', 'max_position_embeddings',
            'rope_theta', 'rms_norm_eps', 'tie_word_embeddings',
            'full_attention_interval', 'linear_num_key_heads',
            'linear_num_value_heads', 'linear_key_head_dim',
            'linear_value_head_dim']:
    val = getattr(config, key, 'N/A')
    print(f"  {key}: {val}")

# ============================================================
# WRITE SUMMARY TO FILE (no tensors, just shapes and stats)
# ============================================================
import json

summary = {
    "model": MODEL_NAME,
    "device": DEVICE,
    "prompt": prompt,
    "input_tokens": inputs['input_ids'].tolist()[0],
    "input_decoded": [tokenizer.decode(t) for t in inputs['input_ids'][0]],
    "config": {
        key: getattr(model.config, key, None)
        for key in ['model_type', 'num_hidden_layers', 'hidden_size',
                     'intermediate_size', 'num_attention_heads',
                     'num_key_value_heads', 'vocab_size',
                     'max_position_embeddings', 'rope_theta',
                     'rms_norm_eps', 'tie_word_embeddings']
    },
    "outputs_type": type(outputs).__name__,
    "outputs_keys": list(outputs.keys()) if hasattr(outputs, 'keys') else [],
    "logits": {
        "shape": list(outputs.logits.shape),
        "dtype": str(outputs.logits.dtype),
        "per_position_entropy_bits": [
            float(-(torch.softmax(outputs.logits[0, pos, :].float(), dim=-1) *
                    torch.log2(torch.softmax(outputs.logits[0, pos, :].float(), dim=-1) + 1e-10)).sum())
            for pos in range(outputs.logits.shape[1])
        ],
        "per_position_top_prediction": [
            tokenizer.decode(outputs.logits[0, pos, :].argmax())
            for pos in range(outputs.logits.shape[1])
        ],
    },
    "hidden_states": {
        "num_layers": len(outputs.hidden_states),
        "layers": [
            {
                "layer": i,
                "label": "embedding" if i == 0 else f"layer_{i}",
                "shape": list(hs.shape),
                "dtype": str(hs.dtype),
                "last_pos_mean": float(hs[0, -1, :].float().mean()),
                "last_pos_std": float(hs[0, -1, :].float().std()),
                "last_pos_norm": float(hs[0, -1, :].float().norm()),
                "last_pos_min": float(hs[0, -1, :].float().min()),
                "last_pos_max": float(hs[0, -1, :].float().max()),
            }
            for i, hs in enumerate(outputs.hidden_states)
        ],
    },
    "attentions": {
        "num_layers": len(outputs.attentions),
        "layers": [
            {
                "layer": i,
                "shape": list(attn.shape),
                "num_heads": attn.shape[1],
                "last_to_first": float(attn[0, :, -1, 0].float().mean()),
                "last_to_self": float(attn[0, :, -1, -1].float().mean()),
                "attn_entropy_bits": float(
                    -(attn[0, :, -1, :].float() *
                      torch.log2(attn[0, :, -1, :].float() + 1e-10)).sum(dim=-1).mean()
                ),
            }
            for i, attn in enumerate(outputs.attentions)
        ],
    },
    "past_key_values": {
        "type": type(outputs.past_key_values).__name__
        if outputs.past_key_values is not None else None,
        "key_cache_layers": len(outputs.past_key_values.key_cache)
        if hasattr(outputs.past_key_values, 'key_cache') else None,
        "layer_map": [
            {
                "layer": i,
                "type": "attention" if k is not None else "gdn",
                "key_shape": list(k.shape) if k is not None else None,
                "value_shape": list(outputs.past_key_values.value_cache[i].shape)
                if k is not None else None,
            }
            for i, k in enumerate(outputs.past_key_values.key_cache)
        ] if hasattr(outputs.past_key_values, 'key_cache') else None,
    },
    "top_predictions": {
        "total_entropy_bits": float(total_entropy),
        "max_entropy_bits": float(torch.log2(torch.tensor(float(probs.shape[0])))),
        "tokens": [
            {
                "rank": rank + 1,
                "token": tokenizer.decode(idx),
                "token_id": idx.item(),
                "probability": float(prob),
                "logprob": float(torch.log(prob)),
            }
            for rank, (prob, idx) in enumerate(zip(top10.values, top10.indices))
        ],
    },
}

out_path = "signal_lab_output.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print(f"Summary written to {out_path}")
print("Done. This is everything accessible from a single forward pass.")
print("=" * 60)
