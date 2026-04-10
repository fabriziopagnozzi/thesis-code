import json
import os
import re

import ollama

OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_DEFAULT_MODEL = os.environ.get('OLLAMA_MODEL', 'gemma4:26b')

ollama_client = ollama.Client(host=OLLAMA_HOST, timeout=300)


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
    messages = [{'role': 'user', 'content': system + '\n\n' + prompt}]

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

    # format='json' enables constrained decoding which suppresses thinking
    fmt: str | None = 'json' if json_mode and not think else None

    if stream:
        content_parts: list[str] = []
        thinking_done = False
        for chunk in ollama_client.chat(
            model=model if model else OLLAMA_DEFAULT_MODEL,
            messages=messages,
            format=fmt,
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

    resp = ollama_client.chat(
        model=model if model else OLLAMA_DEFAULT_MODEL,
        messages=messages,
        format=fmt,
        options=opts,
        think=think,
    )
    content = resp.message.content or ''
    if not content:
        thinking = getattr(resp.message, 'thinking', None)
        if thinking:
            print(f'[ollama] WARNING: empty content, thinking={len(thinking)} chars')
        else:
            print(
                f'[ollama] WARNING: empty content, no thinking. Raw status={getattr(resp, "status_code", "?")}'
            )
    return content


_CODE_FENCE_RE = re.compile(r'^```(?:json)?\s*\n?(.*?)```\s*$', re.DOTALL)
_THINK_TAG_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


def _strip_code_fences(text: str) -> str:
    text = _THINK_TAG_RE.sub('', text).strip()
    m = _CODE_FENCE_RE.match(text)
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
    candidate = text[: last_close + 1].rstrip().rstrip(',') + ']'
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
    stream: bool = False,
) -> dict | list:
    extra_messages: list[dict] = []

    for attempt in range(max_retries + 1):
        full_prompt = prompt
        if extra_messages:
            # For retries, append correction context directly to the prompt
            for msg in extra_messages:
                full_prompt += f'\n\n[{msg["role"].upper()}]: {msg["content"]}'

        raw = generate(
            full_prompt,
            system=system,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_ctx=num_ctx,
            num_predict=num_predict,
            model=model,
            think=think,
            json_mode=True,
            stream=stream,
        )
        text = _strip_code_fences(raw)

        if not text.strip():
            print(f'[generate_json] empty response (attempt {attempt + 1}/{max_retries + 1})')
            if attempt == max_retries:
                raise json.JSONDecodeError('Model returned empty response', text, 0)
            continue

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            first_char = text.strip()[0] if text.strip() else ''
            if first_char not in ('{', '['):
                preview = text[:150].replace('\n', ' ')
                print(f'[generate_json] model returned text instead of JSON: {preview!r}')
                if attempt == max_retries:
                    raise
                extra_messages.append({'role': 'assistant', 'content': raw})
                extra_messages.append(
                    {
                        'role': 'user',
                        'content': 'You must respond with only valid JSON. No prose, no explanation. Output the JSON array now.',
                    }
                )
                continue

            salvaged = _salvage_truncated_json(text)
            if salvaged is not None:
                return salvaged

            print(
                f'[generate_json] bad JSON (attempt {attempt + 1}/{max_retries + 1}, '
                f'{len(text)} chars, error: {exc.msg} at pos {exc.pos}):\n'
                f'  first 80 repr: {text[:80]!r}\n'
                f'  last 80 repr:  {text[-80:]!r}'
            )
            if attempt == max_retries:
                raise

            extra_messages.append({'role': 'assistant', 'content': raw})
            extra_messages.append(
                {
                    'role': 'user',
                    'content': (
                        f'Your previous response was not valid JSON. Parse error: {exc.msg} '
                        f'at position {exc.pos}. Please output only valid JSON, no prose.'
                    ),
                }
            )

    raise RuntimeError('unreachable')
