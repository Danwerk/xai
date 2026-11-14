"""Reusable Matplotlib styles, charts and diagram primitives.

Use ``with plot_style("taltech"):`` for TalTech colours and Verdana (with a
portable fallback). Pass ``overrides=ANALYSIS_SIZES`` for compact notebook
figures. Drawing helpers return artists for further customisation.

Example::

    from utils.plotting import TALTECH_COLORS, horizontal_bars, plot_style

    with plot_style("taltech", overrides={"font.size": 12}):
        fig, ax = plt.subplots(figsize=(10, 5))
        horizontal_bars(ax, ["A", "B"], [12.4, 10.8], TALTECH_COLORS["accent"])
        ax.set_xlabel("MAE")
        fig.tight_layout()

Call ``plt.show()`` to display the figure; export with ``fig.savefig(...)``
only when needed. Importing this module does not change global rcParams.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Circle, Rectangle


# Colours used by the existing TalTech presentation diagrams.
TALTECH_COLORS = {
    "ink": "#342b60",
    "ink_light": "#827b9c",
    "accent": "#e4067e",
    "burgundy": "#aa1352",
    "cyan": "#4dbed2",
    "cyan_dark": "#277786",
    "muted": "#6b687b",
    "line": "#9396b0",
    "border": "#dadae4",
    "grid": "#eeedf2",
    "neutral": "#c8c7d4",
    "background": "white",
}
TALTECH_FONT = ["Verdana", "DejaVu Sans"]

ANALYSIS_STYLE = {
    "figure.dpi": 140,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": "#283747",
    "axes.titlesize": 13,
    "axes.titleweight": "normal",
    "axes.titlepad": 15,
    "axes.labelsize": 11,
    "axes.labelcolor": "#283747",
    "axes.labelpad": 10,
    "axes.edgecolor": "#b4bec9",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "xtick.color": "#536579",
    "ytick.color": "#283747",
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "grid.color": "#e8edf1",
    "grid.linewidth": 0.8,
    "lines.linewidth": 1.8,
}

PRESENTATION_STYLE = {
    "figure.dpi": 140,
    "font.size": 16,
    "axes.titlesize": 19,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

TALTECH_STYLE = {
    **ANALYSIS_STYLE,
    **PRESENTATION_STYLE,
    "font.family": TALTECH_FONT,
    "text.color": TALTECH_COLORS["ink"],
    "axes.labelcolor": TALTECH_COLORS["ink"],
    "axes.edgecolor": TALTECH_COLORS["border"],
    "xtick.color": TALTECH_COLORS["muted"],
    "ytick.color": TALTECH_COLORS["ink"],
    "grid.color": TALTECH_COLORS["grid"],
    "axes.prop_cycle": plt.cycler(color=[
        TALTECH_COLORS[name] for name in ("ink", "accent", "burgundy", "cyan", "line")
    ]),
}

# Reuse the analysis sizes while keeping the TalTech colours and font.
ANALYSIS_SIZES = {
    key: ANALYSIS_STYLE[key]
    for key in ("font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize", "ytick.labelsize")
}


def plot_style(preset="taltech", overrides=None):
    """Return a temporary rc context; optional overrides are Matplotlib rc keys.

    Example: ``with plot_style(overrides={"font.size": 12}): ...``.
    Settings are restored on exit, including when drawing raises an exception.
    """
    styles = {
        "taltech": TALTECH_STYLE,
        "analysis": ANALYSIS_STYLE,
        "presentation": PRESENTATION_STYLE,
    }
    return plt.rc_context({**styles[preset], **(overrides or {})})


def analysis_grid(ax, axis="x", *, color=None, linewidth=0.8):
    """Place a light grid behind the data, using the active style by default."""
    ax.grid(axis=axis, color=color or plt.rcParams["grid.color"], linewidth=linewidth)
    ax.set_axisbelow(True)


def label_bars(ax, bars, digits=2, *, padding=6, size=10, **kwargs):
    """Label horizontal or vertical bars from their values; return text artists."""
    return ax.bar_label(
        bars,
        labels=[f"{value:.{digits}f}" for value in bars.datavalues],
        padding=padding,
        fontsize=size,
        **kwargs,
    )


def horizontal_bars(
    ax, labels, values, colors=None, *, digits=2, height=0.58,
    label_size=10, label_padding=6, headroom=1.24, **kwargs,
):
    """Draw labelled bars with the first item at the top and a visible zero.

    Both positive and negative values are supported. ``headroom`` reserves
    room for value labels; callers can override limits afterwards.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("values must be a non-empty, finite, one-dimensional sequence")
    if len(labels) != len(values):
        raise ValueError("labels and values must have the same length")
    bars = ax.barh(labels, values, color=colors, height=height, **kwargs)
    label_bars(ax, bars, digits, padding=label_padding, size=label_size)
    ax.set_ylim(sorted(ax.get_ylim(), reverse=True))
    low, high = min(float(values.min()), 0), max(float(values.max()), 0)
    ax.set_xlim(low * headroom, max(high * headroom, 1e-9))
    return bars


def analysis_barh(ax, labels, values, colors, digits=2):
    """Notebook analysis bars; return the bar container for customisation."""
    bars = horizontal_bars(ax, labels, values, colors, digits=digits)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", pad=9)
    analysis_grid(ax)
    return bars


def analysis_intervals(ax, values, labels, colors):
    """Draw precomputed intervals without calculating statistical estimates.

    ``values`` must have ``difference``, ``adjusted_low`` and ``adjusted_high``
    columns. Pass a colour string or one colour per row. Return errorbar artists.
    """
    if isinstance(colors, str):
        colors = [colors] * len(values)
    if len(labels) != len(values) or len(colors) != len(values):
        raise ValueError("values, labels and colors must have the same length")
    artists = []
    for position, (row, color) in enumerate(zip(values.itertuples(), colors)):
        artists.append(ax.errorbar(
            row.difference, position,
            xerr=[[row.difference - row.adjusted_low], [row.adjusted_high - row.difference]],
            fmt="o", color=color, markersize=6, elinewidth=1.8, capsize=4, capthick=1.2,
        ))
    ax.axvline(0, color=TALTECH_COLORS["line"], linestyle=(0, (4, 4)), linewidth=1.1)
    ax.set_yticks(range(len(values)), labels)
    ax.set_ylim(len(values) - 0.45, -0.55)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", pad=9)
    analysis_grid(ax)
    return artists


def analysis_dates(ax, maxticks=6, *, minticks=3):
    """Use compact date labels and return the configured locator."""
    locator = AutoDateLocator(minticks=minticks, maxticks=maxticks)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    return locator


def presentation_frame(subtitle="", *, figsize=(16, 9)):
    """Create a white slide-sized figure, optionally with a subtitle."""
    fig = plt.figure(figsize=figsize, facecolor="white")
    if subtitle:
        fig.text(0.06, 0.86, subtitle, fontsize=17, color=TALTECH_COLORS["muted"])
    return fig


def presentation_bars(ax, labels, values, colors, digits=4):
    """Larger labelled bars for slides; set metric labels in the calling code."""
    bars = horizontal_bars(
        ax, labels, values, colors, digits=digits, height=0.57,
        label_size=15, label_padding=8, headroom=1.22,
    )
    analysis_grid(ax)
    return bars


def style_chart_axes(ax, *, grid_axis="x", xpad=9, xsize=12):
    """Apply the shared TalTech chart grid, spines, tick colours and font."""
    analysis_grid(ax, grid_axis, color=TALTECH_COLORS["grid"])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(TALTECH_COLORS["border"])
    ax.tick_params(
        axis="x", length=0, pad=xpad, labelsize=xsize, labelcolor=TALTECH_COLORS["muted"]
    )
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily(TALTECH_FONT)


def format_number(value, signed=False, digits=2):
    """Format a diagram value with a typographic minus and optional plus sign."""
    return format(value, f"{'+' if signed else ''}.{digits}f").replace("-", "−")


def diagram_canvas(*, figsize=(16, 9)):
    """Return (figure, axes) for a diagram in normalised 0–1 coordinates."""
    fig = presentation_frame(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return fig, ax


def diagram_text(ax, x, y, text, size=14, color=None, **kwargs):
    """Add vertically centred TalTech text; return the Text artist."""
    return ax.text(
        x, y, text, fontsize=size, color=color or TALTECH_COLORS["ink"],
        fontfamily=TALTECH_FONT, va="center", zorder=4, **kwargs,
    )


def diagram_box(ax, x, y, width, height, edge=None, linewidth=1.1):
    """Add a white rectangular node; return its patch."""
    box = Rectangle(
        (x, y), width, height, facecolor=TALTECH_COLORS["background"],
        edgecolor=edge or TALTECH_COLORS["border"], linewidth=linewidth, zorder=2,
    )
    ax.add_patch(box)
    return box


def diagram_arrow(
    ax, start, end, color=None, dashed=False, *, linewidth=1.15, mutation_scale=12,
    dash_pattern=(0, (4, 4)),
):
    """Connect two diagram coordinates, optionally with a dashed feedback arrow."""
    return ax.annotate(
        "", xy=end, xytext=start,
        arrowprops={
            "arrowstyle": "-|>", "color": color or TALTECH_COLORS["line"],
            "lw": linewidth, "mutation_scale": mutation_scale,
            "shrinkA": 0, "shrinkB": 0,
            "linestyle": dash_pattern if dashed else "-",
        },
        zorder=1,
    )


def diagram_icon(ax, x, y, kind="clock", color=None):
    """Add a fixed-size clock or slider icon (``clock``, ``parts``, ``single``).

    The icon stays circular even when the figure has a wide aspect ratio.
    Return the AnnotationBbox artist.
    """
    if kind not in {"clock", "parts", "single"}:
        raise ValueError("kind must be 'clock', 'parts' or 'single'")
    color = color or TALTECH_COLORS["ink"]
    drawing = DrawingArea(30, 30, 0, 0)

    def stroke(xs, ys):
        drawing.add_artist(Line2D(
            xs, ys, color=color, linewidth=1.6, solid_capstyle="round",
        ))

    if kind == "clock":
        drawing.add_artist(Circle((15, 15), 10.5, fill=False, edgecolor=color, linewidth=1.6))
        stroke([15, 15, 21], [22, 15, 11])
    else:
        positions = [(6, 10), (15, 21), (24, 15)] if kind == "parts" else [(15, 19)]
        for px, py in positions:
            stroke([px, px], [4, 26])
            drawing.add_artist(Circle(
                (px, py), 3.2, facecolor="white", edgecolor=color, linewidth=1.6,
            ))
    artist = AnnotationBbox(drawing, (x, y), frameon=False, pad=0, zorder=4)
    ax.add_artist(artist)
    return artist
