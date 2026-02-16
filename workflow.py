import random
import uuid


def build_sdxl_workflow(
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg: float,
    seed: int,
    width: int,
    height: int,
    checkpoint_name: str,
    sampler_name: str = "dpmpp_2m",
    scheduler: str = "karras",
    clip_skip: int = -2,
    lora_name: str | None = None,
    lora_strength_model: float = 1.0,
    lora_strength_clip: float = 1.0,
) -> tuple[dict, str, int]:
    """Build a ComfyUI API-format workflow for SDXL text-to-image generation.

    Args:
        prompt: Positive text prompt.
        negative_prompt: Negative text prompt.
        steps: Number of sampling steps.
        cfg: Classifier-free guidance scale.
        seed: RNG seed. -1 means random.
        width: Output image width in pixels.
        height: Output image height in pixels.
        checkpoint_name: Filename of the SDXL checkpoint (e.g. "model.safetensors").
        sampler_name: Sampler algorithm (e.g. "euler", "dpmpp_2m", "dpmpp_2m_sde").
        scheduler: Scheduler type (e.g. "normal", "karras", "exponential").
        clip_skip: CLIP layer to stop at. -1=full, -2=skip 1 layer (SDXL standard).
        lora_name: Optional LoRA filename. If None, no LoRA is applied.
        lora_strength_model: LoRA strength for the diffusion model (0.0-2.0).
        lora_strength_clip: LoRA strength for the CLIP model (0.0-2.0).

    Returns:
        A tuple of (workflow_dict, filename_prefix, actual_seed).
        - workflow_dict: The ComfyUI API-format JSON as a Python dict.
        - filename_prefix: Unique prefix used by SaveImage so we can find the output.
        - actual_seed: The seed that was actually used (resolved if -1).
    """
    actual_seed = seed if seed >= 0 else random.randint(0, 2**63 - 1)
    filename_prefix = f"sdxl_{uuid.uuid4().hex[:12]}"

    workflow = {}

    # Node 1: Load the SDXL checkpoint — outputs: MODEL(0), CLIP(1), VAE(2)
    workflow["1"] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": checkpoint_name,
        },
    }

    # Node 9: CLIP Skip — always present for SDXL/Illustrious XL compatibility
    workflow["9"] = {
        "class_type": "CLIPSetLastLayer",
        "inputs": {
            "clip": ["1", 1],
            "stop_at_clip_layer": clip_skip,
        },
    }

    # Determine which CLIP and MODEL outputs to use for text encoding and sampling
    # If LoRA is enabled, route through LoraLoader; otherwise use CLIP Skip output directly
    if lora_name:
        # Node 8: LoRA Loader (conditional)
        workflow["8"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": lora_name,
                "strength_model": lora_strength_model,
                "strength_clip": lora_strength_clip,
                "model": ["1", 0],
                "clip": ["9", 0],
            },
        }
        clip_source = ["8", 1]
        model_source = ["8", 0]
    else:
        clip_source = ["9", 0]
        model_source = ["1", 0]

    # Node 2: Positive prompt conditioning
    workflow["2"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": prompt,
            "clip": clip_source,
        },
    }

    # Node 3: Negative prompt conditioning
    workflow["3"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": negative_prompt,
            "clip": clip_source,
        },
    }

    # Node 4: Empty latent image at the target resolution
    workflow["4"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": width,
            "height": height,
            "batch_size": 1,
        },
    }

    # Node 5: KSampler — runs the diffusion process
    workflow["5"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": actual_seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": 1.0,
            "model": model_source,
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["4", 0],
        },
    }

    # Node 6: Decode latent to pixel space
    workflow["6"] = {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["5", 0],
            "vae": ["1", 2],
        },
    }

    # Node 7: Save the output image
    workflow["7"] = {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["6", 0],
            "filename_prefix": filename_prefix,
        },
    }

    return workflow, filename_prefix, actual_seed
