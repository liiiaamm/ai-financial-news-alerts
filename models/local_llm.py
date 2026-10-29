"""
This module contains a singleton wrapper for a locally-loaded Qwen2.5-1.5B-Instruct model.

The pipeline is shared by DePlotCaptioner and the alert module so the model weights
are loaded only once per process, avoiding redundant 3 GB loads when both components
are active in the same pipeline run.
"""

# Standard libraries
import threading
import warnings
from typing import Dict, List, Optional


# Module-level singleton state
_lock:            threading.Lock = threading.Lock()
_pipeline:        object | None  = None
_loaded_model_id: str    | None  = None


def _cuda_usable() -> bool:
    """Return True only when CUDA is available and runnable."""

    import torch

    if not torch.cuda.is_available():
        return False
    try:
        _ = torch.zeros(1, device="cuda")
        return True
    except Exception as exc:
        print(f"[LocalLLM] CUDA detected but unusable ({exc}); using CPU.")
        return False


def get_pipeline(
    model_id:  str           = "Qwen/Qwen2.5-1.5B-Instruct",
    cache_dir: Optional[str] = None,
) -> object:
    """
    This function returns the shared text-generation pipeline, loading it once if needed.

    Args:
        model_id:  HuggingFace model ID for the text-generation model.
        cache_dir: Optional directory to store or read model weights;
                   defaults to ~/.cache/huggingface/hub.

    Returns:
        A transformers text-generation pipeline bound to the requested model.
    """

    global _pipeline, _loaded_model_id

    with _lock:
        if _pipeline is not None and _loaded_model_id == model_id:
            return _pipeline

        import torch
        from transformers import pipeline

        cuda_available: bool = _cuda_usable()
        # bfloat16 halves memory vs float32; float16 on CPU is not natively
        # supported by PyTorch and causes a slow software fallback
        dtype = torch.float16 if cuda_available else torch.bfloat16

        model_kwargs: Dict = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
        if cache_dir is not None:
            model_kwargs["cache_dir"] = str(cache_dir)

        # Use device_map="auto" only for usable CUDA + accelerate.
        if cuda_available:
            try:
                import accelerate  # noqa: F401
                extra:        dict = {"device_map": "auto"}
                device_label: str  = "GPU (auto)"
            except ImportError:
                extra        = {"device": 0}
                device_label = "GPU"
        else:
            extra        = {"device": -1}
            device_label = "CPU"

        print(f"[LocalLLM] Loading {model_id} on {device_label}…")
        _pipeline = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs=model_kwargs,
            **extra,
        )
        # Prevent conflict between model's default max_length and max_new_tokens
        _pipeline.model.generation_config.max_length = None
        _loaded_model_id = model_id
        print("[LocalLLM] Model ready.")

    return _pipeline


def generate(
    messages:       List[Dict[str, str]],
    model_id:       str           = "Qwen/Qwen2.5-1.5B-Instruct",
    cache_dir:      Optional[str] = None,
    max_new_tokens: int           = 80,
    temperature:    float         = 0.3,
) -> str:
    """
    This function generates text from a chat message list using the shared local model.

    Args:
        messages:       OpenAI-style chat messages: [{"role": ..., "content": ...}].
        model_id:       HuggingFace model ID for the text-generation model.
        cache_dir:      Optional local cache directory for model weights.
        max_new_tokens: Maximum number of tokens to generate.
        temperature:    Sampling temperature; set to 0 for greedy decoding.

    Returns:
        The generated assistant-turn text as a stripped string.
    """

    generator = get_pipeline(model_id=model_id, cache_dir=cache_dir)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*max_new_tokens.*max_length.*")
        result = generator(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            return_full_text=False,
        )

    # transformers pipeline returns list[list[dict]] or list[dict]
    out = result[0]
    if isinstance(out, list):
        out = out[0]

    return out.get("generated_text", "").strip()
