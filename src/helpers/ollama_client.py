import json
import os
import re

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
    num_ctx: int | None = None,
    num_predict: int | None = None,
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
    if num_ctx is not None:
        opts['num_ctx'] = num_ctx
    if num_predict is not None:
        opts['num_predict'] = num_predict
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
    content = resp.message.content or ''
    if not content:
        thinking = getattr(resp.message, 'thinking', None)
        if thinking:
            print(f'[ollama] WARNING: empty content, thinking={len(thinking)} chars')
        else:
            print(f'[ollama] WARNING: empty content, no thinking. Raw status={getattr(resp, "status_code", "?")}')
    return content


_CODE_FENCE_RE = re.compile(r'^```(?:json)?\s*\n?(.*?)```\s*$', re.DOTALL)


def _strip_code_fences(text: str) -> str:
    m = _CODE_FENCE_RE.match(text.strip())
    return m.group(1).strip() if m else text


def _salvage_truncated_json(text: str) -> list | dict | None:
    """Try to recover a truncated JSON array by finding the last complete element."""
    text = text.strip()
    if not text.startswith('['):
        return None
    # Find the last complete object boundary: "},\n  {" or "}\n]"
    last_close = text.rfind('}')
    if last_close == -1:
        return None
    candidate = text[:last_close + 1].rstrip().rstrip(',') + ']'
    try:
        result = json.loads(candidate)
        print(f'[generate_json] salvaged truncated JSON: kept {len(result)} items')
        return result
    except json.JSONDecodeError:
        return None


def generate_json(
    prompt: str,
    system: str = '',
    temperature: float = 0.1,
    top_p: float | None = None,
    top_k: int | None = None,
    num_ctx: int | None = None,
    num_predict: int | None = None,
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
            num_ctx=num_ctx,
            num_predict=num_predict,
            json_mode=True,
            model=model,
            think=think,
        )
        text = _strip_code_fences(text)
        if not text.strip():
            print(f'[generate_json] empty response (attempt {attempt + 1}/{max_retries + 1})')
            if attempt == max_retries:
                raise json.JSONDecodeError('Model returned empty response', text, 0)
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # If text doesn't start with [ or {, it's natural language — retrying won't help
            first_char = text.strip()[0] if text.strip() else ''
            if first_char not in ('{', '['):
                preview = text[:150].replace('\n', ' ')
                print(f'[generate_json] model returned text instead of JSON: {preview!r}')
                raise
            # Try to salvage truncated JSON before retrying
            salvaged = _salvage_truncated_json(text)
            if salvaged is not None:
                return salvaged
            preview = text[:200] + ('...' if len(text) > 200 else '')
            print(f'[generate_json] bad JSON (attempt {attempt + 1}/{max_retries + 1}, {len(text)} chars): {preview!r}')
            if attempt == max_retries:
                raise
    raise RuntimeError('unreachable')
