"""
Illustrative (not photographic) diagrams for the README's hardware section:
where the IMU sensor mounts on the racket, and what its two sensing axes
(accelerometer + gyroscope) mean for swing analysis. No physical board has
been built yet (README Section 2.7), so these are concept diagrams, clearly
labeled as such -- not renders of, or substitutes for, an actual photo.

Usage:
    python generate_hardware_diagrams.py --out E:/SkillEye/docs/schematics
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, FancyArrowPatch, Arc, Polygon
from matplotlib.path import Path as MplPath

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
BLUE_450 = "#2a78d6"
BLUE_600 = "#184f95"
CRITICAL = "#d03b3b"
GOOD = "#0ca30c"
AMBER = "#c07d0a"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": TEXT_PRIMARY,
    "font.size": 10,
})


def draw_racket(ax, sensor_highlight=False):
    """A simplified, recognizable tennis racket outline (head, throat, handle),
    not a literal product render -- just enough detail to be unambiguous."""
    # Head (strings frame)
    head = Ellipse((0, 2.05), width=2.3, height=3.15, fill=False,
                    edgecolor=TEXT_PRIMARY, linewidth=2.2, zorder=3)
    ax.add_patch(head)

    # Strings (a light grid clipped to the head ellipse)
    for x in np.linspace(-0.95, 0.95, 7):
        line, = ax.plot([x, x], [0.55, 3.55], color=TEXT_SECONDARY, linewidth=0.5, alpha=0.5, zorder=1)
        line.set_clip_path(head)
    for y in np.linspace(0.6, 3.5, 9):
        line, = ax.plot([-1.05, 1.05], [y, y], color=TEXT_SECONDARY, linewidth=0.5, alpha=0.5, zorder=1)
        line.set_clip_path(head)

    # Throat (neck) -- a narrow trapezoid connecting head to handle
    throat = Polygon(
        [(-0.32, 0.15), (0.32, 0.15), (0.16, -0.75), (-0.16, -0.75)],
        closed=True, facecolor=SURFACE, edgecolor=TEXT_PRIMARY, linewidth=2.2, zorder=3,
    )
    ax.add_patch(throat)

    # Handle (grip)
    handle = FancyBboxPatch(
        (-0.22, -2.85), 0.44, 2.15, boxstyle="round,pad=0,rounding_size=0.08",
        facecolor=SURFACE, edgecolor=TEXT_PRIMARY, linewidth=2.2, zorder=3,
    )
    ax.add_patch(handle)

    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-3.2, 3.9)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_sensor_placement(out_path):
    fig, ax = plt.subplots(figsize=(4.6, 6.2))
    draw_racket(ax)

    # Sensor module at the throat/handle junction -- realistic placement:
    # close to the grip, out of the way of the strings and ball contact.
    sensor_xy = (-0.55, -0.55)
    sensor = FancyBboxPatch(
        (sensor_xy[0] - 0.24, sensor_xy[1] - 0.16), 0.48, 0.32,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=BLUE_450, edgecolor=BLUE_600, linewidth=1.5, zorder=5,
    )
    ax.add_patch(sensor)

    ax.annotate(
        "IMU module\n(GY-521/MPU6050 + Seeed XIAO,\nSection 2.7 rev2.1 design)\nconcept placement --\nno physical board mounted yet",
        xy=sensor_xy, xytext=(1.15, -1.1),
        fontsize=8.3, color=TEXT_PRIMARY, ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color=TEXT_SECONDARY, linewidth=1.1),
        bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE, edgecolor=TEXT_SECONDARY, linewidth=0.8),
    )

    ax.set_title("Sensor placement (concept)", fontsize=12, color=TEXT_PRIMARY, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _axis_arrow(ax, origin, dx, dy, color, label, label_xy):
    arrow = FancyArrowPatch(
        origin, (origin[0] + dx, origin[1] + dy),
        arrowstyle="-|>", mutation_scale=16, linewidth=2.2, color=color, zorder=6,
    )
    ax.add_patch(arrow)
    ax.annotate(label, label_xy, fontsize=11, color=color, fontweight="bold", ha="center", va="center")


def _rotation_arc(ax, center, radius, color, label, label_xy):
    arc = Arc(center, radius * 2, radius * 2, angle=0, theta1=20, theta2=290,
              color=color, linewidth=1.6, zorder=6)
    ax.add_patch(arc)
    end_angle = np.radians(290)
    tip = (center[0] + radius * np.cos(end_angle), center[1] + radius * np.sin(end_angle))
    tangent_angle = end_angle + np.pi / 2
    head = FancyArrowPatch(
        (tip[0] - 0.02 * np.cos(tangent_angle), tip[1] - 0.02 * np.sin(tangent_angle)), tip,
        arrowstyle="-|>", mutation_scale=9, linewidth=1.6, color=color, zorder=6,
    )
    ax.add_patch(head)
    ax.annotate(label, label_xy, fontsize=8, color=color, ha="center", va="center")


def fig_sensor_mechanism(out_path):
    fig, (ax, legend_ax) = plt.subplots(1, 2, figsize=(9.5, 5), gridspec_kw={"width_ratios": [1, 1.15]})

    origin = (0, 0)
    chip = FancyBboxPatch((-0.26, -0.18), 0.52, 0.36, boxstyle="round,pad=0.02,rounding_size=0.05",
                           facecolor=BLUE_450, edgecolor=BLUE_600, linewidth=1.5, zorder=5)
    ax.add_patch(chip)
    ax.annotate("IMU", origin, fontsize=8, color="white", ha="center", va="center",
                fontweight="bold", zorder=7)

    # Accelerometer axes (straight arrows), short labels only -- detail lives in the legend panel.
    _axis_arrow(ax, origin, 1.5, 0, CRITICAL, "X", (1.72, 0))
    _axis_arrow(ax, origin, 0, 1.5, GOOD, "Y", (0, 1.75))
    _axis_arrow(ax, origin, -1.05, -1.05, AMBER, "Z", (-1.3, -1.3))

    # Gyroscope rotation glyphs, offset well clear of the axis labels above.
    _rotation_arc(ax, (1.5, -0.75), 0.38, CRITICAL, "roll", (1.5, -1.25))
    _rotation_arc(ax, (0.85, 1.5), 0.38, GOOD, "pitch", (0.85, 2.0))
    _rotation_arc(ax, (-1.75, -0.5), 0.38, AMBER, "yaw", (-1.75, -1.0))

    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.6, 2.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Axes at the mount point", fontsize=12, color=TEXT_PRIMARY, pad=10)

    legend_ax.axis("off")
    legend_ax.set_title("What each axis is meant to help detect", fontsize=12, color=TEXT_PRIMARY, pad=10)
    rows = [
        (CRITICAL, "X -- accel", "racket-face tilt at contact"),
        (CRITICAL, "roll", "rotation about X"),
        (GOOD, "Y -- accel", "swing-plane acceleration"),
        (GOOD, "pitch", "rotation about Y"),
        (AMBER, "Z -- accel", "motion toward/away from the strings"),
        (AMBER, "yaw", "rotation about Z (wrist-snap proxy)"),
    ]
    y0 = 0.92
    for color, tag, desc in rows:
        legend_ax.text(0.02, y0, tag, fontsize=10.5, color=color, fontweight="bold",
                        ha="left", va="center", transform=legend_ax.transAxes)
        legend_ax.text(0.34, y0, desc, fontsize=10, color=TEXT_PRIMARY,
                        ha="left", va="center", transform=legend_ax.transAxes)
        y0 -= 0.14

    fig.suptitle("Sensing mechanism (concept): 3-axis accelerometer + 3-axis gyroscope",
                 fontsize=13, color=TEXT_PRIMARY, y=0.99)
    fig.text(0.5, 0.015,
              "Illustrative -- axis/motion pairings are indicative, not measured; no real sensor data exists yet (Section 2.7).",
              fontsize=8, color=TEXT_SECONDARY, ha="center")
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="E:/SkillEye/docs/schematics")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig_sensor_placement(out / "sensor_placement_concept.png")
    print("wrote", out / "sensor_placement_concept.png")

    fig_sensor_mechanism(out / "sensor_mechanism_concept.png")
    print("wrote", out / "sensor_mechanism_concept.png")


if __name__ == "__main__":
    main()
