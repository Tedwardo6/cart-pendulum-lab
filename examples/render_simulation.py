"""Simulate free motion and save a GIF plus HTML playback controls.

Run: python3 -m examples.render_simulation --links 2 --output runs/double
The HTML is a self-contained animation; no server or internet is required.
"""

import argparse
import os
from pathlib import Path
import re
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cart-pendulum-mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
import numpy as np

from cart_pendulum import PlanarChain
from cart_pendulum.integration import simulate


def offline_controls(html):
    """Replace Matplotlib's externally hosted icon font with text buttons."""
    html = re.sub(r'<link[^>]+font-awesome[^>]*>', '', html)
    labels = {
        'fa-minus': 'Slower', 'fa-fast-backward': 'Start',
        'fa-step-backward': 'Previous', 'fa-play fa-flip-horizontal': 'Reverse',
        'fa-pause': 'Pause', 'fa-play': 'Play', 'fa-step-forward': 'Next',
        'fa-fast-forward': 'End', 'fa-plus': 'Faster',
    }
    for icon, label in labels.items():
        html = html.replace(f'<i class="fa {icon}"></i>', label)
    return html


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links", type=int, default=2, choices=range(1, 5))
    parser.add_argument("--output", type=Path, default=Path("runs/double"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Total chain length stays at 2 m, making comparisons easy to view.
    chain = PlanarChain(1.5, (2.0 / args.links,) * args.links, (0.6,) * args.links)
    angles = np.deg2rad([65, -35, 25, -15][:args.links])
    initial = np.concatenate(([0.0], angles, np.zeros(args.links + 1)))
    times, states = simulate(chain.derivative, initial, duration=12.0, dt=0.002)
    energies = np.array([chain.energy(state) for state in states])
    drift = np.max(np.abs(energies - energies[0]))
    print(f"{args.links} links; 12 seconds; 2 ms physics steps; maximum energy drift {drift:.3e} J")

    # Physics runs at 500 Hz; the drawing samples that trajectory at 25 Hz.
    frames = np.arange(0, len(times), 20)
    positions = np.array([chain.joint_positions(states[k, :args.links + 1]) for k in frames])
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#101927")
    ax.set_facecolor("#101927")
    ax.set_aspect("equal")
    ax.set_xlim(min(-2.4, positions[:, :, 0].min() - .4), max(2.4, positions[:, :, 0].max() + .4))
    ax.set_ylim(-2.35, 0.85)
    ax.set_xlabel("Horizontal position (m)", color="#aab9cc")
    ax.set_ylabel("Height (m)", color="#aab9cc")
    ax.tick_params(colors="#aab9cc")
    for spine in ax.spines.values():
        spine.set_color("#344357")
    ax.grid(color="#344357", alpha=.35)
    ax.axhline(-.2, color="#8795a7", linewidth=2)
    ax.set_title(f"Cart Pendulum Lab  |  {args.links} rods", color="white", loc="left", pad=24, fontsize=18)
    ax.text(0, 1.015, "Released from rest  •  No motor force  •  No friction", transform=ax.transAxes, color="#aab9cc", fontsize=10)
    cart = Rectangle((-.25, -.15), .5, .3, facecolor="#e9eef5", edgecolor="none", zorder=3)
    ax.add_patch(cart)
    colors = ["#56d8d0", "#ffb568", "#b79bff", "#ff829e"]
    rods = [ax.plot([], [], color=colors[i], lw=5, solid_capstyle="round", zorder=4)[0] for i in range(args.links)]
    joints, = ax.plot([], [], "o", color="white", markersize=5, zorder=5)
    clock = ax.text(.02, .04, "", transform=ax.transAxes, color="white", family="monospace")
    fig.tight_layout()

    def draw(frame):
        points = positions[frame]
        cart.set_x(points[0, 0] - .25)
        for i, rod in enumerate(rods):
            rod.set_data(points[i:i+2, 0], points[i:i+2, 1])
        joints.set_data(points[:, 0], points[:, 1])
        clock.set_text(f"t = {times[frames[frame]]:5.2f} s")
        return [cart, *rods, joints, clock]

    animation = FuncAnimation(fig, draw, frames=len(frames), interval=40, blit=True)
    animation.save(args.output / "simulation.gif", writer=PillowWriter(fps=25), dpi=80)
    html = offline_controls(animation.to_jshtml(fps=25, default_mode="once"))
    (args.output / "simulation.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>Cart Pendulum Lab</title>'
        '<style>body{background:#101927;color:#e9eef5;font:16px system-ui;margin:24px} '
        'a{color:#56d8d0}</style></head><body>' + html + '</body></html>', encoding="utf-8")
    draw(0)
    fig.savefig(args.output / "preview.png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
