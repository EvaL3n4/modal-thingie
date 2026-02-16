import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

from config import (
    COMFYUI_BASE_URL,
    COMFYUI_CHECKPOINTS_DIR,
    COMFYUI_LORAS_DIR,
    COMFYUI_PORT,
    MODAL_APP_NAME,
    MODAL_SECRET_NAME,
    MODAL_VOLUME_NAME,
    MODELS_DIR,
)
from workflow import build_sdxl_workflow

# ---------------------------------------------------------------------------
# Modal image: Debian + Python 3.11, ComfyUI installed via comfy-cli
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("comfy-cli==1.5.3", "requests")
    .run_commands("comfy --skip-prompt install --nvidia --version 0.3.71")
    .add_local_file("config.py", "/root/config.py")
    .add_local_file("workflow.py", "/root/workflow.py")
)

volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)
app = modal.App(name=MODAL_APP_NAME, image=image)


@app.cls(
    gpu="L40S",
    volumes={MODELS_DIR: volume},
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
    scaledown_window=300,
    timeout=600,
)
class ComfyUIBackend:
    """Manages a ComfyUI subprocess and exposes generation + model management."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @modal.enter()
    def start_comfyui(self):
        """Launch ComfyUI and wait until it is ready to accept requests."""
        self._sync_model_symlinks()

        subprocess.Popen(
            [
                "comfy", "launch", "--",
                "--listen", "127.0.0.1",
                "--port", str(COMFYUI_PORT),
            ],
        )

        self._wait_for_server()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------
    @modal.method()
    def generate(
        self,
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
    ) -> tuple[bytes, int]:
        """Run an SDXL text-to-image generation and return (image_bytes, actual_seed)."""
        import requests

        workflow, filename_prefix, actual_seed = build_sdxl_workflow(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            cfg=cfg,
            seed=seed,
            width=width,
            height=height,
            checkpoint_name=checkpoint_name,
            sampler_name=sampler_name,
            scheduler=scheduler,
            clip_skip=clip_skip,
            lora_name=lora_name,
            lora_strength_model=lora_strength_model,
            lora_strength_clip=lora_strength_clip,
        )

        # Queue the prompt
        resp = requests.post(
            f"{COMFYUI_BASE_URL}/prompt",
            json={"prompt": workflow},
        )
        resp.raise_for_status()
        body = resp.json()

        if "error" in body:
            raise RuntimeError(f"ComfyUI rejected the workflow: {body['error']}")

        prompt_id = body["prompt_id"]
        image_data = self._wait_for_completion(prompt_id)
        return image_data, actual_seed

    @modal.method()
    def download_model(self, model_url: str) -> str:
        """Download a model from CivitAI and persist it in the volume.

        Args:
            model_url: A full CivitAI download URL or a bare model-version ID.

        Returns:
            The filename of the downloaded checkpoint.
        """
        import requests as req

        api_key = os.environ.get("CIVITAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "CIVITAI_API_KEY is not set. "
                "Create the Modal secret first: "
                "modal secret create civitai-api-key CIVITAI_API_KEY=<key>"
            )

        # Normalise the URL
        if model_url.strip().isdigit():
            download_url = f"https://civitai.com/api/download/models/{model_url.strip()}"
        elif "civitai.com" in model_url:
            download_url = model_url
        else:
            raise ValueError(f"Unrecognised model URL format: {model_url}")

        # CivitAI redirects to S3 which strips headers, so pass the key as
        # a query parameter instead of an Authorization header.
        separator = "&" if "?" in download_url else "?"
        download_url = f"{download_url}{separator}token={api_key}"

        resp = req.get(download_url, stream=True, allow_redirects=True, timeout=600)
        resp.raise_for_status()

        # Extract filename from Content-Disposition, fall back to a safe default
        cd = resp.headers.get("Content-Disposition", "")
        if "filename=" in cd:
            filename = cd.split("filename=")[-1].strip('"').strip("'")
        else:
            slug = model_url.strip().split("/")[-1].split("?")[0]
            filename = f"model_{slug}.safetensors"

        # Stream to disk
        filepath = Path(MODELS_DIR) / filename
        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)

        volume.commit()

        # Make the new model visible to ComfyUI immediately
        target = Path(COMFYUI_CHECKPOINTS_DIR) / filename
        if not target.exists():
            os.symlink(filepath, target)

        print(f"Downloaded and committed: {filename}")
        return filename

    @modal.method()
    def download_lora(self, lora_url: str) -> str:
        """Download a LoRA from CivitAI and persist it in the volume.

        Args:
            lora_url: A full CivitAI download URL or a bare model-version ID.

        Returns:
            The filename of the downloaded LoRA.
        """
        import requests as req

        api_key = os.environ.get("CIVITAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "CIVITAI_API_KEY is not set. "
                "Create the Modal secret first: "
                "modal secret create civitai-api-key CIVITAI_API_KEY=<key>"
            )

        # Normalise the URL
        if lora_url.strip().isdigit():
            download_url = f"https://civitai.com/api/download/models/{lora_url.strip()}"
        elif "civitai.com" in lora_url:
            download_url = lora_url
        else:
            raise ValueError(f"Unrecognised LoRA URL format: {lora_url}")

        # CivitAI redirects to S3 which strips headers, so pass the key as
        # a query parameter instead of an Authorization header.
        separator = "&" if "?" in download_url else "?"
        download_url = f"{download_url}{separator}token={api_key}"

        resp = req.get(download_url, stream=True, allow_redirects=True, timeout=600)
        resp.raise_for_status()

        # Extract filename from Content-Disposition, fall back to a safe default
        cd = resp.headers.get("Content-Disposition", "")
        if "filename=" in cd:
            filename = cd.split("filename=")[-1].strip('"').strip("'")
        else:
            slug = lora_url.strip().split("/")[-1].split("?")[0]
            filename = f"lora_{slug}.safetensors"

        # Ensure loras subdirectory exists
        loras_dir = Path(MODELS_DIR) / "loras"
        loras_dir.mkdir(parents=True, exist_ok=True)

        # Stream to disk
        filepath = loras_dir / filename
        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)

        volume.commit()

        # Make the new LoRA visible to ComfyUI immediately
        target = Path(COMFYUI_LORAS_DIR) / filename
        if not target.exists():
            os.symlink(filepath, target)

        print(f"Downloaded and committed LoRA: {filename}")
        return filename

    @modal.method()
    def list_models(self) -> list[str]:
        """Return filenames of all checkpoints stored in the volume."""
        models_path = Path(MODELS_DIR)
        if not models_path.exists():
            return []
        return sorted(f.name for f in models_path.glob("*.safetensors"))

    @modal.method()
    def list_loras(self) -> list[str]:
        """Return filenames of all LoRAs stored in the volume."""
        loras_path = Path(MODELS_DIR) / "loras"
        if not loras_path.exists():
            return []
        return sorted(f.name for f in loras_path.glob("*.safetensors"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sync_model_symlinks(self):
        """Create symlinks from the volume into ComfyUI's checkpoints and loras directories."""
        # Symlink checkpoints
        os.makedirs(COMFYUI_CHECKPOINTS_DIR, exist_ok=True)
        models_path = Path(MODELS_DIR)
        if models_path.exists():
            for model_file in models_path.glob("*.safetensors"):
                target = Path(COMFYUI_CHECKPOINTS_DIR) / model_file.name
                if not target.exists():
                    os.symlink(model_file, target)

        # Symlink LoRAs
        os.makedirs(COMFYUI_LORAS_DIR, exist_ok=True)
        loras_path = Path(MODELS_DIR) / "loras"
        if loras_path.exists():
            for lora_file in loras_path.glob("*.safetensors"):
                target = Path(COMFYUI_LORAS_DIR) / lora_file.name
                if not target.exists():
                    os.symlink(lora_file, target)

    def _wait_for_server(self, max_retries: int = 60, delay: float = 2.0):
        """Poll ComfyUI's /system_stats endpoint until it responds."""
        for i in range(max_retries):
            try:
                req = urllib.request.Request(f"{COMFYUI_BASE_URL}/system_stats")
                urllib.request.urlopen(req, timeout=5)
                print(f"ComfyUI ready after ~{i * delay:.0f}s")
                return
            except (urllib.error.URLError, ConnectionRefusedError, OSError):
                time.sleep(delay)
        raise RuntimeError(
            f"ComfyUI did not become healthy within {max_retries * delay:.0f}s"
        )

    def _wait_for_completion(self, prompt_id: str, timeout: int = 300) -> bytes:
        """Poll /history until the generation finishes, then fetch the image."""
        import requests

        start = time.time()
        while time.time() - start < timeout:
            resp = requests.get(f"{COMFYUI_BASE_URL}/history/{prompt_id}")
            resp.raise_for_status()
            history = resp.json()

            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                for _node_id, node_output in outputs.items():
                    if "images" in node_output:
                        img_info = node_output["images"][0]
                        img_resp = requests.get(
                            f"{COMFYUI_BASE_URL}/view",
                            params={
                                "filename": img_info["filename"],
                                "subfolder": img_info.get("subfolder", ""),
                                "type": img_info.get("type", "output"),
                            },
                        )
                        img_resp.raise_for_status()
                        return img_resp.content

            time.sleep(1.0)

        raise TimeoutError(f"Generation did not complete within {timeout}s")


# ---------------------------------------------------------------------------
# Quick smoke-test entrypoint: `modal run backend.py`
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    backend = ComfyUIBackend()
    models = backend.list_models.remote()
    print(f"Available models: {models}")
