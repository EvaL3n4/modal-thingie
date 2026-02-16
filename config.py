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
COMFYUI_OUTPUT_DIR = "/root/comfy/ComfyUI/output"
