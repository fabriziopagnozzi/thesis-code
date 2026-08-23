"""Shared rendering choices for figures that are embedded in the thesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PlotRenderingOptions:
    """Visual choices kept consistent across every generated report figure."""

    # Axes titles identify subplots and remain visible independently of this option.
    use_title: bool = False


# Figure captions supply the figure-level interpretation in the thesis, so
# generated assets remain free of duplicate Matplotlib titles by default.
THESIS_PLOT_RENDERING = PlotRenderingOptions(use_title=False)


def set_axis_title(*, axis: Any, title: str, **kwargs: Any) -> None:
    """Render an axes title used to identify a subplot."""
    axis.set_title(title, **kwargs)


def set_figure_title(*, figure: Any, title: str, **kwargs: Any) -> None:
    """Render a figure-level title only when enabled in the global plot settings."""
    if THESIS_PLOT_RENDERING.use_title:
        figure.suptitle(title, **kwargs)


def title_aware_layout_top(*, titled_top: float, untitled_top: float = 0.98) -> float:
    """Return the top layout boundary without reserving unused title space."""
    if THESIS_PLOT_RENDERING.use_title:
        return titled_top
    return untitled_top
