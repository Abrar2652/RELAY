"""
NeurIPS-quality plotting utilities for RELAY experiments.

Enforces a single visual standard across every figure in the paper:
  - No chart titles (titles go in LaTeX captions)
  - Black axis spines, black marker edges
  - Colorblind-safe palette (Wong 2011)
  - Legends that never overlap data (auto-placement + outside-fallback)
  - Serif math, sans-serif labels, 9pt base size (NeurIPS 2026 fits 10pt)
  - PDF + PNG export at 300 DPI
  - Tight bboxes, deterministic output

Import this module before any pyplot work:
    from plotting import setup_style, save_fig, PALETTE, MARKERS
    setup_style()
    ...
    save_fig(fig, 'figures/main_table.pdf')
"""

from __future__ import annotations

import itertools
import os
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Colorblind-safe palette (Wong, Nature Methods 2011) — proven for reviewers
# ---------------------------------------------------------------------------
PALETTE = {
    'blue':          '#0072B2',
    'orange':        '#E69F00',
    'green':         '#009E73',
    'red':           '#D55E00',
    'purple':        '#CC79A7',
    'yellow':        '#F0E442',
    'cyan':          '#56B4E9',
    'black':         '#000000',
    'gray':          '#555555',
}
PALETTE_ORDER = ['blue', 'orange', 'green', 'red', 'purple', 'cyan', 'yellow', 'gray']
PALETTE_LIST = [PALETTE[k] for k in PALETTE_ORDER]

# Marker cycle — distinct shapes so B/W printouts remain readable
MARKERS = ['o', 's', 'D', '^', 'v', 'P', 'X', '*', 'h']

# Line style cycle — backup for B/W reviewers
LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]

# NeurIPS 2026 column widths (inches). textwidth≈5.5", full≈7.0".
WIDTH_SINGLE = 3.4
WIDTH_DOUBLE = 7.0
HEIGHT_SINGLE = 2.3
HEIGHT_DOUBLE = 3.2


def setup_style(base_size: int = 9, use_tex: bool = False) -> None:
    """Install the NeurIPS style on matplotlib's rcParams.

    base_size=9 matches NeurIPS 2026 body text. Set use_tex=True once a LaTeX
    distribution is available for camera-ready; the default renders fine
    without it and avoids build-time failures on research servers.
    """
    mpl.rcParams.update({
        # Fonts
        'font.family':        'sans-serif',
        'font.sans-serif':    ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size':          base_size,
        'axes.titlesize':     base_size,        # unused — no titles — safety only
        'axes.labelsize':     base_size,
        'xtick.labelsize':    base_size - 1,
        'ytick.labelsize':    base_size - 1,
        'legend.fontsize':    base_size - 1,
        'figure.titlesize':   base_size,

        # Math
        'mathtext.fontset':   'stix',
        'text.usetex':        use_tex,

        # Black spines, thin lines — the explicit user requirement
        'axes.edgecolor':     '#000000',
        'axes.linewidth':     0.8,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.grid':          True,
        'grid.color':         '#CCCCCC',
        'grid.linestyle':     '-',
        'grid.linewidth':     0.4,
        'grid.alpha':         0.7,

        # Ticks — outside, small, black
        'xtick.direction':    'out',
        'ytick.direction':    'out',
        'xtick.color':        '#000000',
        'ytick.color':        '#000000',
        'xtick.major.width':  0.8,
        'ytick.major.width':  0.8,
        'xtick.major.size':   3.0,
        'ytick.major.size':   3.0,

        # Lines — thin enough for print, thick enough for screen
        'lines.linewidth':    1.4,
        'lines.markersize':   4.5,
        'lines.markeredgewidth':  0.6,
        'lines.markeredgecolor':  '#000000',

        # Patches (bars / boxes) — always black edges per user rule
        'patch.edgecolor':    '#000000',
        'patch.linewidth':    0.6,
        'patch.force_edgecolor': True,

        # Legends — boxed, small, never transparent so overlap is obvious
        'legend.frameon':         True,
        'legend.framealpha':      0.95,
        'legend.edgecolor':       '#000000',
        'legend.fancybox':        False,
        'legend.borderpad':       0.4,
        'legend.columnspacing':   1.2,
        'legend.handlelength':    1.8,
        'legend.handletextpad':   0.5,

        # Figure
        'figure.dpi':         140,        # screen preview
        'savefig.dpi':        300,        # print-quality export
        'savefig.bbox':       'tight',
        'savefig.pad_inches': 0.02,
        'savefig.format':     'pdf',
        'pdf.fonttype':       42,         # TrueType — passes NeurIPS font check
        'ps.fonttype':        42,

        # Deterministic hatching / no weird defaults
        'hatch.linewidth':    0.5,
    })

    # Replace default color cycle with colorblind-safe palette
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=PALETTE_LIST)


def new_fig(width: float = WIDTH_SINGLE, height: float | None = None,
            ncols: int = 1, nrows: int = 1,
            sharex: bool = False, sharey: bool = False) -> tuple[Figure, Axes | list[Axes]]:
    """Create a correctly-sized figure with the house style."""
    if height is None:
        height = HEIGHT_SINGLE if width <= WIDTH_SINGLE + 0.1 else HEIGHT_DOUBLE
    fig, ax = plt.subplots(
        nrows, ncols,
        figsize=(width, height),
        sharex=sharex, sharey=sharey,
        constrained_layout=True,
    )
    return fig, ax


def place_legend(ax: Axes, *, loc: str = 'best', ncol: int = 1,
                 outside: bool = False, **kwargs) -> None:
    """Place a legend without overlapping data.

    loc='best' lets matplotlib pick the quadrant with the fewest data points.
    outside=True moves the legend to the right of the axes — use this when
    data covers every quadrant.
    """
    kwargs.setdefault('frameon', True)
    kwargs.setdefault('edgecolor', '#000000')
    kwargs.setdefault('fancybox', False)

    if outside:
        # Anchor to the right of the axes; caller should leave room via width
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=ncol, **kwargs)
    else:
        ax.legend(loc=loc, ncol=ncol, **kwargs)


def apply_axes_style(ax: Axes) -> None:
    """Re-apply black spines + clean ticks to an Axes (safety net)."""
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#000000')
        ax.spines[side].set_linewidth(0.8)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors='#000000', width=0.8)


def save_fig(fig: Figure, path: str, *, also_png: bool = True) -> None:
    """Save figure as camera-ready PDF (and a PNG for slides/previews).

    Always saves the PDF (required for NeurIPS). also_png=True also emits a
    300dpi rasterized copy with the same stem for non-LaTeX workflows.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    if not path.endswith('.pdf'):
        raise ValueError(f'save_fig target should end in .pdf: {path}')
    fig.savefig(path)
    if also_png:
        fig.savefig(path[:-4] + '.png')


def style_cycle(n: int) -> list[dict]:
    """Build n distinct (color, marker, linestyle) combinations.

    Each entry is a kwargs dict ready to pass to ax.plot(...). Colors come
    from the colorblind palette, markers and linestyles add redundancy for
    B/W printouts.
    """
    colors = itertools.cycle(PALETTE_LIST)
    markers = itertools.cycle(MARKERS)
    linestyles = itertools.cycle(LINESTYLES)
    return [
        {'color': next(colors), 'marker': next(markers), 'linestyle': next(linestyles),
         'markeredgecolor': '#000000', 'markeredgewidth': 0.6}
        for _ in range(n)
    ]


def bar_kwargs(color_key: str = 'blue') -> dict:
    """Default kwargs for bar charts — enforces black edges per user rule."""
    return {
        'color':         PALETTE[color_key],
        'edgecolor':     '#000000',
        'linewidth':     0.6,
    }


# Register the style by default when imported — safe because idempotent
setup_style()
