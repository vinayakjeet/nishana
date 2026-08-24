"""Serve a trained Nishana adapter behind an OpenAI-compatible endpoint on
Modal, scale-to-zero.

    modal deploy deploy/modal_serve.py

The base model loads once per container; adapters mount from a Volume so adding
an ablation variant is an upload, not a rebuild. This mirrors the LoRAX shape
LoRA Land used to host many adapters over shared base weights: one GPU, several
checkpoints, requests name the adapter they want through the model field.

Cost behaviour lives in deploy/pricing.yaml and nishana/cost/model.py; this
file only owns the serving contract. Tollgate registers the deployed URL as its
specialist route once this endpoint answers.
"""

import modal

APP_NAME = "nishana"
BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct"
ADAPTER_VOLUME = "/adapters"
SCALEDOWN_WINDOW_S = 300

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.8",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(APP_NAME, image=image)


@app.cls(
    gpu="T4",
    scaledown_window=SCALEDOWN_WINDOW_S,
    timeout=600,
    allow_concurrent_inputs=8,
    volumes={ADAPTER_VOLUME: modal.Volume.from_name("nishana-adapters", create_if_missing=True)},
)
@modal.concurrent(max_inputs=16)
class AdapterServer:
    @modal.enter()
    def start(self):
        import subprocess

        self.proc = subprocess.Popen(
            [
                "vllm", "serve", BASE_MODEL,
                "--port", "8000",
                "--enable-lora",
                "--lora-modules",
                "nishana=" + ADAPTER_VOLUME + "/adapter",
                "--max-model-len", "1024",
            ]
        )

    @modal.asgi_app()
    def api(self):
        import httpx
        from fastapi import FastAPI, Request

        web = FastAPI()

        @web.post("/v1/chat/completions")
        async def completions(request: Request):
            body = await request.json()
            async with httpx.AsyncClient(timeout=120) as client:
                return await client.post(
                    "http://127.0.0.1:8000/v1/chat/completions", json=body
                )

        return web
