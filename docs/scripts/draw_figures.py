"""Render the documentation schematics with NumPy and Matplotlib."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

ASSETS = Path(__file__).resolve().parents[1] / "src" / "assets"
THEMES = {
    "light": ("#262626", "#6b6b6b", "#a6482a"),
    "dark": ("#e2e3e5", "#a3a4a8", "#eda98e"),
}


def save(fig, name, theme):
    suffix = "-dark" if theme == "dark" else ""
    fig.savefig(ASSETS / f"docs-{name}{suffix}.png", dpi=200, transparent=True)
    plt.close(fig)


def draw_placement(theme, ink, muted, accent):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.05, top=0.96)
    ax.set(xlim=(-2.65, 2.65), ylim=(-0.7, 1.47))
    ax.axis("off")

    x = np.linspace(-2.3, 2.3, 240)
    scalp = -0.08 * x**2
    ax.plot(x, scalp, color=ink, lw=1.35)
    ax.text(2.03, -0.47, "Scalp", color=ink, ha="center", va="top")

    coil_height = 0.81
    ax.add_patch(
        Rectangle(
            (-0.9, coil_height), 1.8, 0.15, lw=1.4, edgecolor=ink, facecolor=ink, alpha=0.82
        )
    )
    ax.text(0, 1.15, "Coil", color=ink, ha="center", va="center")
    ax.add_patch(
        FancyArrowPatch(
            (0.9, coil_height + 0.075),
            (2.35, coil_height + 0.075),
            arrowstyle="-|>",
            mutation_scale=11,
            lw=1.35,
            color=ink,
        )
    )
    ax.text(1.67, 0.68, "Handle direction", color=ink, ha="center", va="top")

    ax.add_patch(
        FancyArrowPatch(
            (0, 0.06),
            (0, coil_height),
            arrowstyle="<->",
            mutation_scale=10,
            lw=1.35,
            color=accent,
            shrinkA=0,
            shrinkB=0,
        )
    )
    ax.text(-0.13, 0.43, "Coil gap", color=accent, ha="right", va="center")
    ax.scatter([0], [0], color=ink, s=16, zorder=3)
    ax.plot([0, 0], [-0.08, -0.21], color=muted, lw=0.75)
    ax.text(0, -0.29, "Projected centre", color=muted, ha="center", va="top")
    save(fig, "placement", theme)


def draw_focality(theme, ink, muted, accent):
    fig, ax = plt.subplots(figsize=(5.8, 2.65))
    fig.subplots_adjust(left=0.135, right=0.96, bottom=0.25, top=0.96)
    x = np.linspace(0, 1, 1001)
    density = 20 * x * (1 - x) ** 3
    cumulative_volume = 1 - (1 - x) ** 4 * (1 + 4 * x)
    reference = np.interp(0.999, cumulative_volume, x)
    threshold = reference / 2

    ax.fill_between(
        x, 0, density, where=x >= threshold, interpolate=True, facecolor=accent, alpha=0.23
    )
    ax.plot(x, density, color=ink, lw=1.6)
    ax.vlines(threshold, 0, 2.19, color=accent, lw=1.2)
    ax.vlines(reference, 0, 0.77, color=muted, lw=1, linestyle=(0, (3, 3)))
    ax.set(xlim=(0, 1.02), ylim=(0, 2.27), yticks=[])
    ax.set_xticks([threshold, reference], [r"$0.5\,P_{99.9}$", r"$P_{99.9}$"])
    ax.tick_params(axis="x", length=0, pad=8)
    ax.get_xticklabels()[0].set_color(accent)
    ax.get_xticklabels()[1].set_color(muted)
    ax.set_xlabel(r"Electric field magnitude, $|E|$", labelpad=7, color=ink)
    ax.set_ylabel("Volume density", labelpad=10, color=ink)
    ax.spines[["top", "right"]].set_visible(False)
    for name in ("bottom", "left"):
        ax.spines[name].set_color(muted)
        ax.spines[name].set_linewidth(0.75)
    save(fig, "focality", theme)


if __name__ == "__main__":
    plt.switch_backend("Agg")
    plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})
    for name, palette in THEMES.items():
        draw_placement(name, *palette)
        draw_focality(name, *palette)
