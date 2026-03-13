import argparse
import time
import json
import os
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import dotenv
import numpy as np
from colony.model.prompt import Prompt
from colony.model.g_profile import (
    VALID_G_FUNCTIONS,
    build_attention_scales_from_spec,
    printable_scales,
)

# ensure HF_TOKEN loaded from .env_development
dotenv.load_dotenv(".env.development")
dotenv.load_dotenv(".env")

def resolve_device(requested_device: str | None = None) -> str:
    """
    Resolve runtime device.

    Priority:
    1) explicit argument
    2) COLONY_DEVICE env var
    3) auto-detect (cuda > mps > cpu)
    """
    env_device = os.getenv("COLONY_DEVICE")
    raw_value = (requested_device or env_device or "auto").strip().lower()

    if raw_value == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    valid = {"cuda", "mps", "cpu"}
    if raw_value not in valid:
        valid_str = ", ".join(sorted(valid | {"auto"}))
        raise ValueError(f"Invalid device '{raw_value}'. Expected one of: {valid_str}.")

    if raw_value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Device 'cuda' requested, but CUDA is not available in this environment.")
    if raw_value == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Device 'mps' requested, but MPS is not available in this environment.")
    return raw_value


def resolve_hf_token() -> str | None:
    """
    Resolve a Hugging Face token from common env var names.
    """
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.getenv(key)
        if value:
            return value
    return None


DEVICE = resolve_device()
MODEL_NAME_4B = "Qwen/Qwen3.5-4B-Base"
MODEL_NAME_2B = "Qwen/Qwen3.5-2B-Base"
MODEL_NAME_0_8B = "Qwen/Qwen3.5-0.8B-Base"
MODEL_NAME_9B = "Qwen/Qwen3.5-9B-Base"
MODEL_NAME_35B = "Qwen/Qwen3.5-35B-A3B-Base"
MODEL_NAME_OLMO = "allenai/Olmo-Hybrid-7B"
DATA_DIR = Path("data")

def get_attention_layer_indices(model) -> list[int]:
    return [i for i in range(len(model.model.layers)) if (i + 1) % 4 == 0]


def attention_scaler_hook(scale: float):
    """
    A forward hook function that scales the attention output of layer `idx` by the corresponding element of `g`.
    Note that there is a bit of trickiness here in that not each layer is an attention layer. The g_vec vector is 
    padded with zeros for non-attention layers, and thus we need only match the idx to the corresponding element of g_vec.

    Args:
        idx: The index of the layer to scale.
        g_vec: The g vector to scale the attention output by.

    Returns:
        A forward hook function that scales the attention output of layer `idx` by the corresponding element of `g`.
    """
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            scaled = output[0] * scale
            return (scaled,) + output[1:]
        elif isinstance(output, torch.Tensor):
            return output * scale
        else:
            output[0] = output[0] * scale
            return output
    return hook_fn

def load_model_and_tokenizer(model_name: str, device: str = DEVICE):
    """
    Load a model and tokenizer from the Hugging Face model hub.

    Args:
        model_name: The name of the model to load.
        device: The device to load the model on.
    """
    if model_name not in [MODEL_NAME_0_8B, MODEL_NAME_2B, MODEL_NAME_4B, MODEL_NAME_9B, MODEL_NAME_35B, MODEL_NAME_OLMO]: # ready for a refactor!
        raise ValueError(f"Invalid model: {model_name}")

    device = resolve_device(device)
    hf_token = resolve_hf_token()

    print(f"Device: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"CUDA GPU: {gpu_name}")
    if not hf_token:
        print(
            "Warning: no HF token found in HF_TOKEN/HUGGINGFACE_HUB_TOKEN. "
            "If model download fails, set one of these env vars."
        )
    print(f"Loading {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            dtype=torch.float16,
            device_map=device,
            attn_implementation="eager",
            output_hidden_states=True,
            output_attentions=True
        )
    except Exception as exc:
        if not hf_token:
            raise RuntimeError(
                "Failed to load model and no Hugging Face token was found. "
                "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN on this machine."
            ) from exc
        raise
    _model.eval()
    print(f"Model loaded: {_model.config.num_hidden_layers} layers, hidden_size={_model.config.hidden_size}")
    return _model, tokenizer

def _resolve_path(path_or_name: str) -> Path:
    path = Path(path_or_name)
    if path.is_file():
        return path
    data_path = DATA_DIR / path_or_name
    if data_path.is_file():
        return data_path
    return path


def _load_prompt_entries(json_path: Path) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    if not isinstance(data, list):
        raise ValueError(f"Prompt catalog must be a list: {json_path}")
    return data


def _prompt_from_catalog(json_path: Path, prompt_id: str) -> Prompt:
    entries = _load_prompt_entries(json_path)
    for entry in entries:
        if isinstance(entry, dict) and entry.get("id") == prompt_id:
            return Prompt.from_dict(entry, data_dir=json_path.parent, source=str(json_path))
    raise ValueError(f"Prompt id '{prompt_id}' not found in {json_path}")


def _all_prompt_catalogs() -> list[Path]:
    if not DATA_DIR.is_dir():
        return []
    return sorted(DATA_DIR.glob("prompts*.json"))


def resolve_prompt(prompt_arg: str) -> Prompt:
    """
    Resolve a prompt argument into a Prompt object.

    Supported forms:
    - direct string prompt
    - text file path or file name in `data/`
    - prompt id from any `data/prompts*.json` catalog (for example `short0`)
    - explicit catalog selector `prompts_short.json:short0`
    """
    # Explicit selector: "<catalog_path_or_name>:<prompt_id>"
    if ":" in prompt_arg:
        left, right = prompt_arg.split(":", 1)
        maybe_catalog = _resolve_path(left)
        if maybe_catalog.is_file() and maybe_catalog.suffix == ".json":
            return _prompt_from_catalog(maybe_catalog, right)

    # Direct file path / data-relative file path
    resolved_path = _resolve_path(prompt_arg)
    if resolved_path.is_file():
        if resolved_path.suffix == ".json":
            entries = _load_prompt_entries(resolved_path)
            if len(entries) == 1:
                entry = entries[0]
                if not isinstance(entry, dict):
                    raise ValueError(f"Invalid prompt entry in {resolved_path}")
                return Prompt.from_dict(entry, data_dir=resolved_path.parent, source=str(resolved_path))
            raise ValueError(
                f"Prompt catalog {resolved_path} contains multiple prompts. "
                "Use '<catalog>:<id>' (for example 'prompts_short.json:short0')."
            )

        with open(resolved_path, "r", encoding="utf-8") as file_handle:
            return Prompt(
                id=resolved_path.name,
                prompt_text=file_handle.read().strip(),
                prompt_file=str(resolved_path),
                source=str(resolved_path),
            )

    # Catalog prompt id lookup across all prompt catalogs.
    prompt_matches = []
    for catalog_path in _all_prompt_catalogs():
        for entry in _load_prompt_entries(catalog_path):
            if isinstance(entry, dict) and entry.get("id") == prompt_arg:
                prompt_matches.append(
                    Prompt.from_dict(entry, data_dir=catalog_path.parent, source=str(catalog_path))
                )

    if len(prompt_matches) == 1:
        return prompt_matches[0]
    if len(prompt_matches) > 1:
        sources = ", ".join(sorted({match.source for match in prompt_matches if match.source}))
        raise ValueError(
            f"Prompt id '{prompt_arg}' is ambiguous across catalogs: {sources}. "
            "Use '<catalog>:<id>' to disambiguate."
        )

    # Fallback: literal prompt string.
    return Prompt.from_text(prompt_arg, prompt_id="direct_prompt")


def read_prompt(prompt_arg):
    """
    Backwards-compatible helper that returns only the prompt text.
    """
    return resolve_prompt(prompt_arg).prompt_text

def run_model_pass(
    model,
    tokenizer,
    prompt,
    g_attention_scales: np.ndarray | list[float],
    device=DEVICE,
    prompt_id=None,
    target_token_id=None,
    baseline_logits=None,
    return_raw_logits=False,
    return_verbose=False,
):
    start_time = time.perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    attn_layers = get_attention_layer_indices(model)
    scales = np.asarray(g_attention_scales, dtype=float)
    if scales.ndim != 1 or scales.size != len(attn_layers):
        raise ValueError(
            f"g_attention_scales length ({scales.size}) must equal "
            f"attention layer count ({len(attn_layers)})."
        )
    hooks = []

    for idx, scale in zip(attn_layers, scales.tolist()):
        layer = model.model.layers[idx]
        handle = layer.register_forward_hook(attention_scaler_hook(scale))
        hooks.append(handle)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
        
    for h in hooks:
        h.remove()
        
    elapsed_time = time.perf_counter() - start_time
    
    final_logits = outputs.logits[0, -1, :].float()
    probs = torch.softmax(final_logits, dim=-1)
    
    if return_verbose:
        top_logits, top_indices = torch.topk(final_logits, 15)
        top_tokens = [tokenizer.decode(idx) for idx in top_indices]
    
    target_token = None
    target_rank = None
    target_prob = None
    if target_token_id is not None:
        target_token = tokenizer.decode(target_token_id)
        target_prob = probs[target_token_id].item()
        target_rank = (probs > probs[target_token_id]).sum().item() + 1
        
    final_entropy_bits = -(probs * torch.log2(probs + 1e-10)).sum().item()
    
    mean_entropy_bits = None
    if return_verbose:
        all_logits = outputs.logits[0, :, :].float()
        all_probs = torch.softmax(all_logits, dim=-1)
        mean_entropy_bits = -(all_probs * torch.log2(all_probs + 1e-10)).sum(dim=-1).mean().item()
    
    kl_from_baseline = None
    if baseline_logits is not None:
        if not isinstance(baseline_logits, torch.Tensor):
            baseline_logits = torch.tensor(baseline_logits).to(device)
        baseline_probs = torch.softmax(baseline_logits.float(), dim=-1)
        kl_from_baseline = F.kl_div(
            torch.log(probs + 1e-10),
            baseline_probs,
            reduction='sum'
        ).item()
        
    attn_entropy = []
    if return_verbose and hasattr(outputs, 'attentions') and outputs.attentions is not None:
        for attn in outputs.attentions:
            last_token_attn = attn[0, :, -1, :].float()
            ent = -(last_token_attn * torch.log2(last_token_attn + 1e-10)).sum(dim=-1)
            attn_entropy.append(ent.tolist())
            
    result = {
        "prompt_id": prompt_id if prompt_id else (prompt[:20] + "..."),
        "g_attention_scales": printable_scales(scales),
        "attention_layer_indices": attn_layers,
        "target_token": target_token,
        "target_rank": target_rank,
        "target_prob": target_prob,
        "final_entropy_bits": final_entropy_bits,
        "kl_from_baseline": kl_from_baseline,
        "elapsed_time": elapsed_time
    }
    
    if return_verbose:
        result["top_k_logits"] = top_logits.tolist()
        result["top_k_indices"] = top_indices.tolist()
        result["top_k_tokens"] = top_tokens
        result["mean_entropy_bits"] = mean_entropy_bits
        result["attn_entropy_per_head_final"] = attn_entropy
    
    if return_raw_logits:
        result["_raw_logits"] = final_logits

    return result


def score_target_sequence(
    model,
    tokenizer,
    prompt: str,
    target_text: str,
    g_attention_scales: np.ndarray | list[float],
    device=DEVICE,
):
    """
    Score an entire target continuation token-by-token under teacher forcing.

    Returns sequence-level metrics plus per-token diagnostics:
    - `target_seq_logprob`: sum of log probabilities over target tokens
    - `target_avg_logprob`: mean log probability per target token
    - `target_geo_mean_prob`: exp(mean log probability)
    - `target_token_*`: per-token ids/text/logprobs/probs/ranks
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    if not target_ids:
        return None

    if not prompt_ids:
        raise ValueError("Target sequence scoring requires a non-empty prompt.")

    all_ids = prompt_ids + target_ids
    input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)

    attn_layers = get_attention_layer_indices(model)
    scales = np.asarray(g_attention_scales, dtype=float)
    if scales.ndim != 1 or scales.size != len(attn_layers):
        raise ValueError(
            f"g_attention_scales length ({scales.size}) must equal "
            f"attention layer count ({len(attn_layers)})."
        )

    hooks = []
    for idx, scale in zip(attn_layers, scales.tolist()):
        layer = model.model.layers[idx]
        handle = layer.register_forward_hook(attention_scaler_hook(scale))
        hooks.append(handle)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    for handle in hooks:
        handle.remove()

    logits = outputs.logits[0].float()  # [seq_len, vocab]
    prompt_len = len(prompt_ids)

    token_logprobs = []
    token_probs = []
    token_ranks = []
    for offset, token_id in enumerate(target_ids):
        # Logits at position t predict token at position t+1.
        logit_pos = prompt_len + offset - 1
        token_logits = logits[logit_pos]
        token_log_probs = F.log_softmax(token_logits, dim=-1)
        lp = token_log_probs[token_id]
        prob = torch.exp(lp)
        rank = (token_logits > token_logits[token_id]).sum().item() + 1
        token_logprobs.append(lp.item())
        token_probs.append(prob.item())
        token_ranks.append(rank)

    seq_logprob = float(np.sum(token_logprobs))
    avg_logprob = float(np.mean(token_logprobs))
    geo_mean_prob = float(np.exp(avg_logprob))

    return {
        "target_text": target_text,
        "target_num_tokens": len(target_ids),
        "target_token_ids": target_ids,
        "target_tokens": [tokenizer.decode(token_id) for token_id in target_ids],
        "target_token_logprobs": token_logprobs,
        "target_token_probs": token_probs,
        "target_token_ranks": token_ranks,
        "target_seq_logprob": seq_logprob,
        "target_avg_logprob": avg_logprob,
        "target_geo_mean_prob": geo_mean_prob,
        "target_first_token_rank": token_ranks[0],
        "target_first_token_prob": token_probs[0],
    }

def _parse_csv_floats(raw_value: str | None) -> list[float] | None:
    if raw_value is None:
        return None
    values = [piece.strip() for piece in raw_value.split(",") if piece.strip()]
    if not values:
        return None
    return [float(piece) for piece in values]


def run_model(
    prompt_source,
    model_key: str = "0_8B",
    device=DEVICE,
    g_spec: dict[str, Any] | None = None,
):
    if model_key not in ["0_8B", "2B", "4B", "9B"]:
        raise ValueError(f"Invalid model key: {model_key}")

    model_name = {
        "0_8B": MODEL_NAME_0_8B,
        "2B": MODEL_NAME_2B,
        "4B": MODEL_NAME_4B,
        "9B": MODEL_NAME_9B,
    }[model_key]

    model, tokenizer = load_model_and_tokenizer(model_name, device)
    attn_layers = get_attention_layer_indices(model)
    resolved_g_spec = g_spec or {"g_function": "constant", "g_params": {"value": 1.0}}
    g_scales = build_attention_scales_from_spec(resolved_g_spec, attention_slots=len(attn_layers))
        
    prompt = resolve_prompt(prompt_source)
    
    summary = run_model_pass(
        model,
        tokenizer,
        prompt.prompt_text,
        g_scales,
        device=device,
        prompt_id=prompt.id,
        return_verbose=True,
    )
    
    config_dict = {
        key: getattr(model.config, key, None)
        for key in ['model_type', 'num_hidden_layers', 'hidden_size',
                     'intermediate_size', 'num_attention_heads',
                     'num_key_value_heads', 'vocab_size']
    }
    summary["model"] = model_name
    summary["device"] = device
    summary["config"] = config_dict
    summary["g_spec"] = resolved_g_spec
    
    out_path = "signal_lab_output.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nSummary written to {out_path}")
    print(f"Elapsed time: {summary['elapsed_time']:.3f}s")
    print(
        "Top prediction "
        f"(g_function={resolved_g_spec.get('g_function')}, "
        f"scales={summary['g_attention_scales']}): "
        f"index {summary['top_k_indices'][0]} logit {summary['top_k_logits'][0]:.3f}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signal Lab: exploring model internals via transformers.")
    parser.add_argument("--prompt", type=str, default="The color with the shortest wavelength is", help="Path or filename to a prompt file in data/, or a direct string prompt.")
    parser.add_argument("--model-key", type=str, default="0_8B", help="Model to use. Please enter 0_8B, 2B, 4B, or 9B.")
    parser.add_argument(
        "--g-function",
        type=str,
        default="constant",
        choices=sorted(VALID_G_FUNCTIONS),
        help="g profile function family.",
    )
    parser.add_argument(
        "--g",
        type=float,
        default=1.0,
        help="Constant value shortcut when --g-function=constant.",
    )
    parser.add_argument(
        "--g-vector",
        type=str,
        default=None,
        help="Comma-separated control-point values for --g-function=control_points.",
    )
    parser.add_argument(
        "--g-params-json",
        type=str,
        default=None,
        help="JSON object with extra g function params (for example '{\"slope\": 0.4, \"intercept\": 1.0}').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: auto (default), cuda, mps, or cpu. Also supports COLONY_DEVICE env var.",
    )
    args = parser.parse_args()

    g_params = json.loads(args.g_params_json) if args.g_params_json else {}
    if not isinstance(g_params, dict):
        raise ValueError("--g-params-json must decode to a JSON object.")

    if args.g_function == "constant" and "value" not in g_params:
        g_params["value"] = args.g

    g_spec = {
        "g_function": args.g_function,
        "g_params": g_params,
    }
    g_vector = _parse_csv_floats(args.g_vector)
    if g_vector is not None:
        g_spec["g_vector"] = g_vector

    run_model(args.prompt, args.model_key, device=resolve_device(args.device), g_spec=g_spec)
