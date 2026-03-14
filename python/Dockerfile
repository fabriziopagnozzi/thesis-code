FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS build

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tcl-dev \
    tk-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --frozen --no-install-project

COPY . .
RUN uv sync --no-dev --frozen

FROM python:3.14-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libtcl8.6 \
    libtk8.6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /app /app

ENV PATH="/app/.venv/bin:$PATH"
ENV TCL_LIBRARY=/usr/share/tcltk/tcl8.6
ENV TK_LIBRARY=/usr/share/tcltk/tk8.6

CMD ["python", "main.py"]
