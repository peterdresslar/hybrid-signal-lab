import argparse
import os
import time
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import dotenv
import numpy as np

# ensure HF_TOKEN loaded from .env_development
dotenv.load_dotenv(".env.development")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_NAME_4B = "Qwen/Qwen3.5-4B-Base"
MODEL_NAME_2B = "Qwen/Qwen3.5-2B-Base"
MODEL_NAME_0_8B = "Qwen/Qwen3.5-0.8B-Base"
QWEN_LAYERS = 24

def generate_g_vector_qwen35(g_value_or_vector: float | np.ndarray = 1.0):
    """
    Generate a g vector of length `num_layers`. Can send an argument for single-value g, or a vector of length 6.

    If a single value is passed as an argument, the output will place that value every fourth position in the output vector.
    If a vector is passed as an argument, the output will place the values in the vector at the corresponding positions.
    
    Args:
        g_value_or_vector: The g value or vector to generate.

    Returns:
        A numpy array of length 24.
    """
    output_vec= np.zeros(QWEN_LAYERS)

    if isinstance(g_value_or_vector, float):
        output_vec[3::4] = g_value_or_vector
    elif isinstance(g_value_or_vector, np.ndarray) and len(g_value_or_vector) == 6:
        output_vec[3::4] = g_value_or_vector
    else:
        raise ValueError(f"Invalid type for g: {type(g_value_or_vector)}")
    return output_vec

def g_vec_as_printable_array(g_vec: np.ndarray):
    """
    Convert a g vector to a printable array without all the extra zeroes. Should return a list of exactly 6 floats.
    """
    return g_vec[3::4].tolist()

def attention_scaler_hook(idx: int, g_vec: np.ndarray):
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
            scaled = output[0] * g_vec[idx]
            return (scaled,) + output[1:]
        elif isinstance(output, torch.Tensor):
            return output * g_vec[idx]
        else:
            output[0] = output[0] * g_vec[idx]
            return output
    return hook_fn

def load_model_and_tokenizer(model_name: str, device: str = DEVICE):
    """
    Load a model and tokenizer from the Hugging Face model hub.

    Args:
        model_name: The name of the model to load.
        device: The device to load the model on.
    """
    if model_name not in [MODEL_NAME_0_8B, MODEL_NAME_2B, MODEL_NAME_4B]:
        raise ValueError(f"Invalid model: {model_name}")

    print(f"Device: {device}")
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map=device,
        attn_implementation="eager",
        output_hidden_states=True,
        output_attentions=True
    )
    _model.eval()
    print(f"Model loaded: {_model.config.num_hidden_layers} layers, hidden_size={_model.config.hidden_size}")
    return _model, tokenizer

def read_prompt(prompt_arg):
    if os.path.isfile(prompt_arg):
        with open(prompt_arg, "r", encoding="utf-8") as f:
            return f.read().strip()
    data_path = os.path.join("data", prompt_arg)
    if os.path.isfile(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return prompt_arg

def run_model_pass(model, tokenizer, prompt, g_vec, device=DEVICE, prompt_id=None, target_token_id=None, baseline_logits=None, return_raw_logits=False, return_verbose=False):
    start_time = time.perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    attn_layers = [i for i in range(len(model.model.layers)) if (i + 1) % 4 == 0]
    hooks = []
    
    for idx in attn_layers:
        layer = model.model.layers[idx]
        handle = layer.register_forward_hook(attention_scaler_hook(idx, g_vec))
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
        "g_vector": g_vec_as_printable_array(g_vec),
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

def run_model(prompt_source, g_value_or_vector, model_key: str = "0_8B", device=DEVICE):
    if model_key not in ["0_8B", "2B", "8B"]:
        raise ValueError(f"Invalid model key: {model_key}")

    model_name = {
        "0_8B": MODEL_NAME_0_8B,
        "2B": MODEL_NAME_2B,
        "4B": MODEL_NAME_4B
    }[model_key]

    model, tokenizer = load_model_and_tokenizer(model_name, device)

    g_vec = generate_g_vector_qwen35(g_value_or_vector)
        
    prompt = read_prompt(prompt_source)
    prompt_id = prompt_source if os.path.isfile(prompt_source) or os.path.isfile(os.path.join("data", prompt_source)) else "direct_prompt"
    
    summary = run_model_pass(model, tokenizer, prompt, g_vec, device=device, prompt_id=prompt_id, return_verbose=True)
    
    config_dict = {
        key: getattr(model.config, key, None)
        for key in ['model_type', 'num_hidden_layers', 'hidden_size',
                     'intermediate_size', 'num_attention_heads',
                     'num_key_value_heads', 'vocab_size']
    }
    summary["model"] = model_name
    summary["device"] = device
    summary["config"] = config_dict
    
    out_path = "signal_lab_output.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nSummary written to {out_path}")
    print(f"Elapsed time: {summary['elapsed_time']:.3f}s")
    print(f"Top prediction (g={g_vec_as_printable_array(g_vec)}): index {summary['top_k_indices'][0]} logit {summary['top_k_logits'][0]:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signal Lab: exploring model internals via transformers.")
    parser.add_argument("--prompt", type=str, default="The color with the shortest wavelength is", help="Path or filename to a prompt file in data/, or a direct string prompt.")
    parser.add_argument("--g", type=float, default=1.0, help="Attention scaler")
    parser.add_argument("--model-key", type=str, default="0_8B", help="Model to use. Please enter 0_8B, 2B, or 4B.")
    args = parser.parse_args()
    
    run_model(args.prompt, args.g, args.model_key)
