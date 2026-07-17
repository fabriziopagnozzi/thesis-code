import io
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from experiments.medical_dataset_gen.utils.constants import COLOR_CODES, RESET, ColorLike
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths


def setup_logging(paths: MedicalDatasetGenPaths, run_id: str | None = None) -> None:
    main = sys.modules['__main__']
    script_name = Path(main.__file__ if main.__file__ else f'unknown_script_{uuid4()}').stem
    suffix = run_id or datetime.now().strftime('%Y%m%dT%H%M%S_%f')
    log_path = paths.logs_dir / f'{script_name}_{suffix}.log'

    class _Tee(io.TextIOBase):
        def __init__(self, filepath: Path):
            self._terminal = sys.stdout
            self._file = open(filepath, 'w')  # noqa: SIM115

        def write(self, msg: str) -> int:
            self._terminal.write(msg)
            self._file.write(msg)
            return len(msg)

        def flush(self) -> None:
            self._terminal.flush()
            self._file.flush()

    sys.stdout = _Tee(log_path)


def _normalize_color_name(color: str) -> str:
    return color.strip().lower().replace('-', '_').replace(' ', '_')


def _resolve_color(color: ColorLike) -> int:
    if isinstance(color, int):
        if not 0 <= color <= 255:
            raise ValueError(f'Terminal color code must be between 0 and 255, got {color}')
        return color

    key = _normalize_color_name(color)

    if key not in COLOR_CODES:
        raise ValueError(
            f'Unknown color {color!r}. Use a named color, an int from 0-255, '
            f"or a string like 'c208'."
        )

    return COLOR_CODES[key]


def colored(color: ColorLike, text: str) -> str:
    code = _resolve_color(color)
    return f'\033[38;5;{code}m{text}{RESET}'


def colorprint(color: ColorLike, text: str) -> None:
    print(colored(color, text))


def bg_colored(
    color: ColorLike,
    text: str,
    *,
    fg: ColorLike | None = None,
    bold: bool = False,
) -> str:
    bg_code = _resolve_color(color)

    codes: list[str] = []

    if bold:
        codes.append('1')

    if fg is not None:
        fg_code = _resolve_color(fg)
        codes.append(f'38;5;{fg_code}')

    codes.append(f'48;5;{bg_code}')

    return f'\033[{";".join(codes)}m{text}{RESET}'


def bg_colorprint(
    color: ColorLike,
    text: str,
    *,
    fg: ColorLike | None = None,
    bold: bool = False,
) -> None:
    print(bg_colored(color, text, fg=fg, bold=bold))
