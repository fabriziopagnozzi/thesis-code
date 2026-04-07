import json
import os

import ollama

OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_DEFAULT_MODEL = os.environ.get('OLLAMA_MODEL', 'qwen3.5:35b')

_client = ollama.Client(host=OLLAMA_HOST)


def generate(
    prompt: str,
    system: str = '',
    temperature: float = 0.1,
    top_p: float | None = None,
    top_k: int | None = None,
    json_mode: bool = False,
    model: str | None = None,
    think: bool = False,
    stream: bool = False,
) -> str:
    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.append({'role': 'user', 'content': prompt})

    opts: dict = {'temperature': temperature}
    if top_p is not None:
        opts['top_p'] = top_p
    if top_k is not None:
        opts['top_k'] = top_k
    if think:
        opts['think'] = True

    if stream:
        content_parts: list[str] = []
        thinking_done = False
        for chunk in _client.chat(
            model=model if model else OLLAMA_DEFAULT_MODEL,
            messages=messages,
            format='json' if json_mode else '',
            options=opts,
            think=think,
            stream=True,
        ):
            thinking_token = chunk.message.thinking or ''
            if thinking_token:
                print(thinking_token, end='', flush=True)
            content_token = chunk.message.content or ''
            if content_token and not thinking_done:
                print('\n------- RESPONSE -------')
                thinking_done = True
            if content_token:
                print(content_token, end='', flush=True)
            content_parts.append(content_token)
        print()
        return ''.join(content_parts)

    resp = _client.chat(
        model=model if model else OLLAMA_DEFAULT_MODEL,
        messages=messages,
        format='json' if json_mode else '',
        options=opts,
        think=think,
    )
    return resp.message.content or ''


def generate_json(
    prompt: str,
    system: str = '',
    temperature: float = 0.1,
    top_p: float | None = None,
    top_k: int | None = None,
    max_retries: int = 2,
    model: str | None = None,
    think: bool = False,
) -> dict | list:
    for attempt in range(max_retries + 1):
        text = generate(
            prompt,
            system=system,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            json_mode=True,
            model=model,
            think=think,
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if attempt == max_retries:
                raise
            temperature = min(temperature + 0.1, 0.5)
    raise RuntimeError('unreachable')
