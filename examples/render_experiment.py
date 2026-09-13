"""Replay saved validation trajectories side by side, without training or API calls."""

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cart-pendulum-mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
import numpy as np

from cart_pendulum import PlanarChain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads((args.run / "history.json").read_text())
    if not records:
        parser.error("The run has no completed trials.")
    args.output.mkdir(parents=True, exist_ok=False)
    panels = []
    fig, axes = plt.subplots(1, len(records), figsize=(6 * len(records), 5), squeeze=False)
    fig.patch.set_facecolor("#101927")
    colors = ["#56d8d0", "#ffb568", "#b79bff", "#ff829e"]
    for ax, record in zip(axes[0], records):
        trial = args.run / record["trial"]
        config = json.loads((trial / "config.json").read_text())["environment"]
        n = config["n_links"]
        chain = PlanarChain(config["cart_mass"], (config["total_length"] / n,) * n,
                            (config["rod_mass"],) * n)
        data = np.loadtxt(trial / "validation_trajectory.csv", delimiter=",", skiprows=1, ndmin=2)
        points = np.array([chain.joint_positions(row[1:n+2]) for row in data])
        episode = record["validation"]["episodes"][0]
        ax.set_facecolor("#101927")
        ax.set_aspect("equal")
        extent = max(config["track_limit"] + .4, float(np.abs(points[:, :, 0]).max()) + .4)
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-.65, config["total_length"] + .55)
        ax.axhline(-.16, color="#8795a7")
        ax.axvline(-config["track_limit"], color="#ff829e", ls=":", alpha=.5)
        ax.axvline(config["track_limit"], color="#ff829e", ls=":", alpha=.5)
        ax.tick_params(colors="#aab9cc")
        ax.set_xlabel("Cart position (m)", color="#aab9cc")
        for spine in ax.spines.values():
            spine.set_color("#344357")
        sizes = " → ".join(map(str, record["network"]["hidden_sizes"]))
        ax.set_title(f"{record['trial']}  |  {sizes}\nValidation seed {episode['seed']}", color="white", pad=18)
        cart = Rectangle((0, -.15), .4, .3, color="#e9eef5")
        ax.add_patch(cart)
        rods = [ax.plot([], [], "o-", color=colors[i % len(colors)], lw=4, ms=5)[0] for i in range(n)]
        label = ax.text(.03, .96, "", transform=ax.transAxes, va="top", color="white", fontsize=10)
        panels.append((data, points, episode, cart, rods, label))
    fig.suptitle("Saved controller evaluations · real-time playback", color="white", fontsize=16)
    fig.text(.5, .025, "Each panel freezes when its episode ends. These are individual episodes, not average scores.",
             ha="center", color="#aab9cc", fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .92))
    fps = 25
    end = max(p[0][-1, 0] for p in panels)

    def draw(t):
        for data, points, episode, cart, rods, label in panels:
            # Replay recorded states; do not integrate or invent post-failure motion.
            index = min(np.searchsorted(data[:, 0], t, side="right") - 1, len(data) - 1)
            index = max(0, index)
            p = points[index]
            cart.set_x(p[0, 0] - .2)
            for i, rod in enumerate(rods):
                rod.set_data(p[i:i+2, 0], p[i:i+2, 1])
            ended = t >= data[-1, 0]
            status = episode["end_reason"].replace("_", " ") if ended else "balancing"
            label.set_text(f"t = {data[index, 0]:.2f} s  |  {status}\nLast applied force: {data[index, -1]:+.1f} N")

    frames = np.arange(0, end + 1.04, 1 / fps)
    animation = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    animation.save(args.output / "comparison.gif", writer=PillowWriter(fps=fps), dpi=90)
    draw(end)
    fig.savefig(args.output / "final-frame.png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(args.output / "comparison.gif")


if __name__ == "__main__":
    main()
