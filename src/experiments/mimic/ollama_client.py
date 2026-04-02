import json
import os

import ollama

OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'qwen3:8b')

_client = ollama.Client(host=OLLAMA_HOST)


def generate(
    prompt: str,
    system: str = '',
    temperature: float = 0.1,
    json_mode: bool = False,
) -> str:
    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.append({'role': 'user', 'content': prompt})

    resp = _client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        format='json' if json_mode else '',
        options={'temperature': temperature},
    )
    return resp.message.content or ''


def generate_json(
    prompt: str,
    system: str = '',
    temperature: float = 0.1,
    max_retries: int = 2,
) -> dict | list:
    for attempt in range(max_retries + 1):
        text = generate(prompt, system=system, temperature=temperature, json_mode=True)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if attempt == max_retries:
                raise
            temperature = min(temperature + 0.1, 0.5)
    raise RuntimeError('unreachable')
