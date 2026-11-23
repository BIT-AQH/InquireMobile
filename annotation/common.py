"""Shared helpers for the annotation scripts (OpenAI-compatible chat API)."""

import base64
import json
import os
import time
from typing import Optional

from openai import OpenAI

DEFAULT_MODEL = os.environ.get("ANNOTATION_MODEL", "gpt-4o-0806")


def get_client() -> OpenAI:
    """Build a client from ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``."""
    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def chat(system_prompt: str, user_prompt: str, image_path: Optional[str] = None,
         model: str = DEFAULT_MODEL, json_mode: bool = False, temperature: float = 0.1,
         max_retries: int = 3, retry_delay: float = 3.0):
    """Query the model, optionally with one screenshot. Returns str, or dict if ``json_mode``."""
    content = [{"type": "text", "text": user_prompt}]
    if image_path is not None:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(image_path)}"},
        })
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}

    for attempt in range(max_retries):
        try:
            response = get_client().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=temperature,
                **kwargs,
            )
            text = response.choices[0].message.content
            return json.loads(text) if json_mode else text.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[API failed] image={image_path} error={e}")
                raise
            time.sleep(retry_delay)


def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)
