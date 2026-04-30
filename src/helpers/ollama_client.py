import json
import os
import re

import ollama

OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_DEFAULT_MODEL = os.environ.get('OLLAMA_MODEL', 'gemma4-31b-text')

ollama_client = ollama.Client(host=OLLAMA_HOST)


def generate(
    prompt: str,
    system: str = '',
    model: str | None = None,
    think: bool = False,
    stream: bool = False,
    **kwargs,
) -> str:
    used_model = model or OLLAMA_DEFAULT_MODEL
    messages = [{'role': 'user', 'content': system + '\n\n' + prompt}]

    prompt_chars = sum(len(m['content']) for m in messages)
    print(f'[ollama] {used_model} | ~{prompt_chars // 4} tokens | think={think}', flush=True)

    if stream:
        content_parts: list[str] = []
        thinking_done = False

        for chunk in ollama_client.chat(
            model=used_model, messages=messages, options=kwargs, think=think, stream=True
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

    reply = ollama_client.chat(model=used_model, messages=messages, options=kwargs, think=think)

    content = reply.message.content or ''
    print(f'[ollama] response: {len(content)} chars', flush=True)
    if not content:
        thinking = getattr(reply.message, 'thinking', None)
        if thinking:
            print(f'[ollama] WARNING: empty content, thinking={len(thinking)} chars')
        else:
            print(f'[ollama] WARNING: empty content, no thinking. Raw={reply!r}')
    return content


def generate_json(
    prompt: str,
    system: str = '',
    max_retries: int = 2,
    **kwargs,
) -> dict | list:
    extra_messages: list[dict] = []

    for attempt in range(max_retries + 1):
        full_prompt = prompt
        if extra_messages:
            for msg in extra_messages:
                full_prompt += f'\n\n[{msg["role"].upper()}]: {msg["content"]}'

        raw = generate(full_prompt, system=system, **kwargs)
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
                        'content': 'You must respond with only valid JSON. No prose, no explanation. Output the JSON now.',
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


_CODE_FENCE_RE = re.compile(r'```(?:json)?\s*\n?(.*?)```', re.DOTALL)
_THINK_TAG_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


def _strip_code_fences(text: str) -> str:
    text = _THINK_TAG_RE.sub('', text).strip()
    matches = _CODE_FENCE_RE.findall(text)
    return matches[-1].strip() if matches else text


def _salvage_truncated_json(text: str) -> list | dict | None:
    """Try to recover a truncated JSON array by finding the last complete element."""
    text = text.strip()
    if not text.startswith('['):
        return None
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
