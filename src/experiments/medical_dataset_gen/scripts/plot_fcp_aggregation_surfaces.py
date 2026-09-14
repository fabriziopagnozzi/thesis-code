#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

GridFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]

# magma, inferno, plasma, viridis, cividis, twilight, twilight_shifted, turbo, berlin, managua, vanimo, Blues, BrBG, BuGn, BuPu, CMRmap, GnBu, Greens, Greys, OrRd, Oranges, PRGn, PiYG, PuBu, PuBuGn, PuOr, PuRd, Purples, RdBu, RdGy, RdPu, RdYlBu, RdYlGn, Reds, Spectral, Wistia, YlGn, YlGnBu, YlOrBr, YlOrRd, afmhot, autumn, binary, bone, brg, bwr, cool, coolwarm, copper, cubehelix, flag, gist_earth, gist_gray, gist_heat, gist_ncar, gist_rainbow, gist_stern, gist_yarg, gnuplot, gnuplot2, gray, hot, hsv, jet, nipy_spectral, ocean, pink, prism, rainbow, seismic, spring, summer, terrain, winter, Accent, Dark2, Paired, Pastel1, Pastel2, Set1, Set2, Set3, tab10, tab20, tab20b, tab20c, grey, gist_grey, gist_yerg, Grays


def _harmonic_mean(precision: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    numerator = 2.0 * precision * coverage
    denominator = precision + coverage
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )


def _score_functions() -> list[tuple[str, GridFunction]]:
    return [
        (r'Product (FCP): $PC$', lambda precision, coverage: precision * coverage),
        (
            r'Geometric mean: $\sqrt{PC}$',
            lambda precision, coverage: np.sqrt(precision * coverage),
        ),
        (r'Harmonic mean: $\frac{2PC}{P+C}$', _harmonic_mean),
        (
            r'Arithmetic mean: $\frac{P+C}{2}$',
            lambda precision, coverage: (precision + coverage) / 2.0,
        ),
        (
            r'Bottleneck: $\min(P,C)$',
            lambda precision, coverage: np.minimum(precision, coverage),
        ),
    ]


def _style_heatmap_axis(axis: Axes) -> None:
    ticks = np.linspace(0.0, 1.0, 5)
    axis.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xticks=ticks,
        yticks=ticks,
        xlabel='Precision, $P$',
        ylabel='Facet coverage, $C$',
    )
    axis.set_aspect('equal')


def _plot_diagonal(axis: Axes) -> None:
    values = np.linspace(0.0, 1.0, 1000)
    axis.plot(values, values**2, color='#154c79', linewidth=3.0, label=r'Product: $t^2$')
    axis.plot(
        values,
        values,
        color='#d95f02',
        linewidth=2.7,
        linestyle='--',
        label=r'All four alternatives: $t$',
    )
    axis.scatter(
        [0.5, 0.5],
        [0.25, 0.5],
        color=['#154c79', '#d95f02'],
        s=38,
        zorder=3,
    )
    axis.annotate(
        '0.25',
        (0.5, 0.25),
        xytext=(7, -13),
        textcoords='offset points',
        fontsize=10,
    )
    axis.annotate(
        '0.50',
        (0.5, 0.5),
        xytext=(7, 3),
        textcoords='offset points',
        fontsize=10,
    )
    axis.set(
        title=r'Balanced inputs: $P=C=t$',
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xticks=np.linspace(0.0, 1.0, 5),
        yticks=np.linspace(0.0, 1.0, 5),
        xlabel='Component value, $t$',
        ylabel='Aggregate score',
    )
    axis.set_aspect('equal')
    axis.grid(color='#d7d7d7', linewidth=0.7)
    axis.legend(loc='upper left', frameon=False, fontsize=10)


def render(
    output_directory: Path, resolution: int = 501, dpi: int = 600, palette: str = 'viridis'
) -> Path:
    """Render the comparison to a high-resolution PNG file."""
    component_values = np.linspace(0.0, 1.0, resolution)
    precision, coverage = np.meshgrid(component_values, component_values)

    plt.rcParams.update(
        {
            'font.family': 'DejaVu Sans',
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(11.8, 8.2), layout='constrained')  # type:ignore
    flat_axes = list(axes.flat)
    heatmaps = []

    contour_levels = np.arange(0.2, 1.0, 0.2)
    for axis, (title, score_function) in zip(flat_axes[:5], _score_functions(), strict=True):
        scores = score_function(precision, coverage)
        heatmap = axis.imshow(
            scores,
            origin='lower',
            extent=(0.0, 1.0, 0.0, 1.0),
            vmin=0.0,
            vmax=1.0,
            cmap=palette,
            interpolation='nearest',
        )
        contours = axis.contour(
            precision,
            coverage,
            scores,
            levels=contour_levels,
            colors='white',
            linewidths=1.0,
            alpha=0.9,
        )
        axis.clabel(contours, inline=True, fmt='%.1f', fontsize=9)
        axis.set_title(title)
        _style_heatmap_axis(axis)
        heatmaps.append(heatmap)

    _plot_diagonal(flat_axes[-1])
    colorbar = figure.colorbar(
        heatmaps[0],
        ax=flat_axes[:5],
        location='bottom',
        orientation='horizontal',
        shrink=0.82,
        aspect=55,
        pad=0.035,
    )
    colorbar.set_label('Aggregate score')
    colorbar.set_ticks(np.linspace(0.0, 1.0, 6))

    output_directory.mkdir(parents=True, exist_ok=True)
    png_path = output_directory / 'fcp_aggregation_surfaces.png'
    figure.savefig(png_path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-directory',
        type=Path,
        default=(
            Path(__file__).resolve().parents[5] / 'thesis-writing' / 'reports' / 'figures_custom'
        ),
        help='Directory for fcp_aggregation_surfaces.png.',
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=501,
        help='Number of grid points per component axis (default: 501).',
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=600,
        help='PNG export resolution in dots per inch (default: 600).',
    )
    parser.add_argument(
        '--palette',
        type=str,
        default='viridis',
    )
    arguments = parser.parse_args()
    if arguments.resolution < 2:
        parser.error('--resolution must be at least 2')
    if arguments.dpi < 72:
        parser.error('--dpi must be at least 72')

    png_path = render(
        arguments.output_directory, arguments.resolution, arguments.dpi, arguments.palette
    )
    print(png_path)


if __name__ == '__main__':
    main()
