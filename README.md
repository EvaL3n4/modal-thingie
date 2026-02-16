# modal-thingie

SDXL image generation powered by ComfyUI on Modal with a local Gradio UI.

## Architecture

```
Gradio (local)  ──.remote()──>  Modal container (L40S GPU)
                                  └── ComfyUI subprocess (localhost:8188)
                                        └── SDXL checkpoint (from Modal Volume)
```

- **Backend** (`backend.py`): A Modal class that runs ComfyUI as a subprocess on an L40S GPU. Handles model downloads from CivitAI and image generation via ComfyUI's workflow API.
- **Frontend** (`app.py`): A Gradio 6.5.1 Blocks interface that runs locally and calls the Modal backend with `.remote()`.
- **Workflow** (`workflow.py`): Builds ComfyUI API-format workflow JSON dynamically from user parameters.

## Prerequisites

- Python 3.11+
- A [Modal](https://modal.com) account with the CLI authenticated (`modal token set`)
- A [CivitAI](https://civitai.com) API key (for downloading models)

## Setup

### 1. Install local dependencies

```bash
pip install -r requirements.txt
```

### 2. Create the CivitAI secret in Modal

```bash
modal secret create civitai-api-key CIVITAI_API_KEY=your_civitai_api_key_here
```

### 3. Deploy the backend

```bash
modal deploy backend.py
```

This builds the container image (first deploy takes a few minutes), registers the `ComfyUIBackend` class, and makes it callable via `.remote()`.

### 4. Run the Gradio UI

```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

## Usage

1. **Download a model**: Expand "Model Settings", paste a CivitAI model URL or version ID, and click "Download Model". The checkpoint is saved to a persistent Modal Volume.
2. **Select the model**: Pick it from the "Model Checkpoint" dropdown (click "Refresh Models" if it doesn't appear).
3. **Generate**: Enter a prompt, adjust parameters, and click "Generate". The first request after a cold start takes ~60-90 seconds (container boot + model loading). Subsequent requests are much faster.

## Quick smoke test (no UI)

```bash
modal run backend.py
```

This starts a container, launches ComfyUI, and prints the list of available models.

## Project structure

```
.gitignore
requirements.txt    Local Python dependencies
config.py           Shared constants (resolutions, paths, Modal config)
workflow.py         ComfyUI API-format workflow builder
backend.py          Modal backend (GPU inference + model management)
app.py              Local Gradio frontend
```
