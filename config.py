# Resolution presets: label -> (width, height)
# All ~1 megapixel, which is optimal for SDXL
RESOLUTION_PRESETS = {
    "1024x1024 (Square)": (1024, 1024),
    "896x1152 (Portrait)": (896, 1152),
    "1152x896 (Landscape)": (1152, 896),
    "768x1344 (Tall Portrait)": (768, 1344),
    "1344x768 (Wide Landscape)": (1344, 768),
}

# Default generation parameters
DEFAULT_STEPS = 30
DEFAULT_CFG = 7.0
DEFAULT_SEED = -1  # -1 means random
DEFAULT_RESOLUTION = "1024x1024 (Square)"

# Sampler and scheduler options
SAMPLERS = [
    "euler", "euler_ancestral", "euler_cfg_pp",
    "heun", "heunpp2",
    "dpm_2", "dpm_2_ancestral",
    "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
    "dpmpp_sde", "dpmpp_sde_gpu",
    "lms", "dpm_fast", "dpm_adaptive",
]

SCHEDULERS = [
    "normal", "karras", "exponential", "sgm_uniform",
    "simple", "ddim_uniform", "ays", "ays+", "kl_optimal",
]

DEFAULT_SAMPLER = "dpmpp_2m"
DEFAULT_SCHEDULER = "karras"

# CLIP Skip configuration
DEFAULT_CLIP_SKIP = -2  # SDXL/Illustrious XL standard

# LoRA configuration
DEFAULT_LORA_STRENGTH_MODEL = 1.0
DEFAULT_LORA_STRENGTH_CLIP = 1.0

# Modal configuration
MODAL_APP_NAME = "sdxl-comfyui-backend"
MODAL_CLS_NAME = "ComfyUIBackend"
MODAL_VOLUME_NAME = "comfyui-models"
MODAL_SECRET_NAME = "civitai-api-key"

# Paths inside the Modal container
COMFYUI_PORT = 8188
COMFYUI_BASE_URL = f"http://127.0.0.1:{COMFYUI_PORT}"
MODELS_DIR = "/models"
COMFYUI_CHECKPOINTS_DIR = "/root/comfy/ComfyUI/models/checkpoints"
COMFYUI_LORAS_DIR = "/root/comfy/ComfyUI/models/loras"
COMFYUI_OUTPUT_DIR = "/root/comfy/ComfyUI/output"
