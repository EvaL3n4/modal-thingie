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

    Returns:
        A tuple of (workflow_dict, filename_prefix, actual_seed).
        - workflow_dict: The ComfyUI API-format JSON as a Python dict.
        - filename_prefix: Unique prefix used by SaveImage so we can find the output.
        - actual_seed: The seed that was actually used (resolved if -1).
    """
    actual_seed = seed if seed >= 0 else random.randint(0, 2**63 - 1)
    filename_prefix = f"sdxl_{uuid.uuid4().hex[:12]}"

    workflow = {
        # Load the SDXL checkpoint — outputs: MODEL(0), CLIP(1), VAE(2)
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": checkpoint_name,
            },
        },
        # Positive prompt conditioning
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["1", 1],
            },
        },
        # Negative prompt conditioning
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["1", 1],
            },
        },
        # Empty latent image at the target resolution
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        },
        # KSampler — runs the diffusion process
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": actual_seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
        },
        # Decode latent to pixel space
        "6": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["5", 0],
                "vae": ["1", 2],
            },
        },
        # Save the output image
        "7": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["6", 0],
                "filename_prefix": filename_prefix,
            },
        },
    }

    return workflow, filename_prefix, actual_seed
