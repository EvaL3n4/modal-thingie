import io

import gradio as gr
import modal
from PIL import Image

from config import (
    DEFAULT_CFG,
    DEFAULT_CLIP_SKIP,
    DEFAULT_LORA_STRENGTH_CLIP,
    DEFAULT_LORA_STRENGTH_MODEL,
    DEFAULT_RESOLUTION,
    DEFAULT_SAMPLER,
    DEFAULT_SCHEDULER,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    MODAL_APP_NAME,
    MODAL_CLS_NAME,
    RESOLUTION_PRESETS,
    SAMPLERS,
    SCHEDULERS,
)

# ---------------------------------------------------------------------------
# Connect to the deployed Modal backend
# ---------------------------------------------------------------------------
ComfyUIBackend = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLS_NAME)
backend = ComfyUIBackend()


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------
def generate_image(
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg: float,
    seed: int,
    resolution: str,
    model_name: str,
    sampler_name: str,
    scheduler: str,
    clip_skip: int,
    lora_name: str,
    lora_strength_model: float,
    lora_strength_clip: float,
):
    if not prompt.strip():
        raise gr.Error("Prompt cannot be empty.")
    if not model_name:
        raise gr.Error("No model selected. Download a model first.")

    width, height = RESOLUTION_PRESETS[resolution]

    # Convert empty LoRA dropdown to None
    lora_param = lora_name if lora_name and lora_name != "None" else None

    try:
        image_bytes, actual_seed = backend.generate.remote(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=int(steps),
            cfg=float(cfg),
            seed=int(seed),
            width=width,
            height=height,
            checkpoint_name=model_name,
            sampler_name=sampler_name,
            scheduler=scheduler,
            clip_skip=int(clip_skip),
            lora_name=lora_param,
            lora_strength_model=float(lora_strength_model),
            lora_strength_clip=float(lora_strength_clip),
        )
    except Exception as exc:
        raise gr.Error(f"Generation failed: {exc}")

    image = Image.open(io.BytesIO(image_bytes))
    return image, str(actual_seed)


def download_model(model_url: str):
    if not model_url.strip():
        raise gr.Error("Model URL / version ID is required.")

    try:
        filename = backend.download_model.remote(model_url=model_url.strip())
        models = backend.list_models.remote()
        return (
            gr.update(value=f"Downloaded: {filename}"),
            gr.update(choices=models, value=filename),
        )
    except Exception as exc:
        raise gr.Error(f"Download failed: {exc}")


def refresh_models():
    try:
        models = backend.list_models.remote()
        return gr.update(choices=models, value=models[0] if models else None)
    except Exception:
        return gr.update(choices=[], value=None)


def download_lora(lora_url: str):
    if not lora_url.strip():
        raise gr.Error("LoRA URL / version ID is required.")

    try:
        filename = backend.download_lora.remote(lora_url=lora_url.strip())
        loras = backend.list_loras.remote()
        # Add "None" as first option
        lora_choices = ["None"] + loras
        return (
            gr.update(value=f"Downloaded: {filename}"),
            gr.update(choices=lora_choices, value=filename),
        )
    except Exception as exc:
        raise gr.Error(f"Download failed: {exc}")


def refresh_loras():
    try:
        loras = backend.list_loras.remote()
        lora_choices = ["None"] + loras
        return gr.update(choices=lora_choices, value="None")
    except Exception:
        return gr.update(choices=["None"], value="None")


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="SDXL Image Generator") as demo:
    gr.Markdown("# SDXL Image Generator\nPowered by ComfyUI on Modal (L40S)")

    with gr.Row():
        # ---- Left column: controls ----
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="A photorealistic landscape at golden hour...",
                lines=3,
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                placeholder="blurry, low quality, deformed, watermark...",
                lines=2,
            )

            with gr.Row():
                steps = gr.Slider(
                    minimum=1, maximum=100, value=DEFAULT_STEPS,
                    step=1, label="Steps",
                )
                cfg = gr.Slider(
                    minimum=1.0, maximum=20.0, value=DEFAULT_CFG,
                    step=0.5, label="CFG Scale",
                )

            with gr.Row():
                seed = gr.Number(
                    value=DEFAULT_SEED,
                    label="Seed (-1 = random)",
                    precision=0,
                )
                resolution = gr.Dropdown(
                    choices=list(RESOLUTION_PRESETS.keys()),
                    value=DEFAULT_RESOLUTION,
                    label="Resolution",
                )

            model_dropdown = gr.Dropdown(
                choices=[], label="Model Checkpoint", interactive=True,
            )
            refresh_btn = gr.Button("Refresh Models", size="sm")

            # Advanced Settings
            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    sampler = gr.Dropdown(
                        choices=SAMPLERS,
                        value=DEFAULT_SAMPLER,
                        label="Sampler",
                    )
                    scheduler = gr.Dropdown(
                        choices=SCHEDULERS,
                        value=DEFAULT_SCHEDULER,
                        label="Scheduler",
                    )

                clip_skip = gr.Slider(
                    minimum=-24,
                    maximum=-1,
                    value=DEFAULT_CLIP_SKIP,
                    step=1,
                    label="CLIP Skip",
                    info="SDXL/Illustrious XL: use -2 (default). More negative = more abstract.",
                )

                lora_dropdown = gr.Dropdown(
                    choices=["None"],
                    value="None",
                    label="LoRA (Optional)",
                    interactive=True,
                )
                refresh_loras_btn = gr.Button("Refresh LoRAs", size="sm")

                with gr.Row():
                    lora_strength_model = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=DEFAULT_LORA_STRENGTH_MODEL,
                        step=0.05,
                        label="LoRA Strength (Model)",
                    )
                    lora_strength_clip = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=DEFAULT_LORA_STRENGTH_CLIP,
                        step=0.05,
                        label="LoRA Strength (CLIP)",
                    )

            generate_btn = gr.Button("Generate", variant="primary", size="lg")
            actual_seed_display = gr.Textbox(
                label="Actual Seed Used", interactive=False,
            )

        # ---- Right column: output ----
        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Image", type="pil")

    # ---- Model management (collapsed by default) ----
    with gr.Accordion("Model Settings", open=False):
        gr.Markdown(
            "**Setup:** Create the CivitAI secret in Modal before downloading:\n\n"
            "```\nmodal secret create civitai-api-key CIVITAI_API_KEY=your_key\n```"
        )

        gr.Markdown("### Download Model Checkpoint")
        model_url = gr.Textbox(
            label="CivitAI Model URL or Version ID",
            placeholder="https://civitai.com/api/download/models/12345 or just 12345",
        )
        download_btn = gr.Button("Download Model", variant="secondary")
        download_status = gr.Textbox(label="Download Status", interactive=False)

        gr.Markdown("### Download LoRA")
        lora_url = gr.Textbox(
            label="CivitAI LoRA URL or Version ID",
            placeholder="https://civitai.com/api/download/models/67890 or just 67890",
        )
        download_lora_btn = gr.Button("Download LoRA", variant="secondary")
        lora_download_status = gr.Textbox(label="LoRA Download Status", interactive=False)

    # ---- Wire events ----
    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt, negative_prompt, steps, cfg, seed, resolution, model_dropdown,
            sampler, scheduler, clip_skip,
            lora_dropdown, lora_strength_model, lora_strength_clip,
        ],
        outputs=[output_image, actual_seed_display],
    )
    refresh_btn.click(fn=refresh_models, outputs=[model_dropdown])
    refresh_loras_btn.click(fn=refresh_loras, outputs=[lora_dropdown])
    download_btn.click(
        fn=download_model,
        inputs=[model_url],
        outputs=[download_status, model_dropdown],
    )
    download_lora_btn.click(
        fn=download_lora,
        inputs=[lora_url],
        outputs=[lora_download_status, lora_dropdown],
    )

    # Auto-refresh on page load
    demo.load(fn=refresh_models, outputs=[model_dropdown])
    demo.load(fn=refresh_loras, outputs=[lora_dropdown])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
