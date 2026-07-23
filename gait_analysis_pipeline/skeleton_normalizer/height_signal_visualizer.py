from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt


class HeightSignalVisualizer:
    """
    Visualizes raw vs smoothed height signal.
    """

    @staticmethod
    def plot(
        raw: np.ndarray,
        smoothed: np.ndarray,
        title: str = "Height Signal",
        fps: float | None = None,
        save_svg: bool = True,
        output_path: str = "height_signal.svg",
    ) -> None:

        raw = np.asarray(raw).reshape(-1)
        smoothed = np.asarray(smoothed).reshape(-1)

        num_frames = len(raw)

        # Time axis
        if fps:
            x = np.arange(num_frames) / fps
            x_label = "Time (seconds)"
        else:
            x = np.arange(num_frames)
            x_label = "Frame Index"

        # Larger figure for paper
        fig, ax = plt.subplots(figsize=(10, 4))

        # Signals
        ax.plot(
            x,
            raw,
            label="Raw Height",
            alpha=0.5,
            linewidth=1.5,
            color="steelblue",
        )

        ax.plot(
            x,
            smoothed,
            label="Smoothed Height",
            linewidth=2.5,
            color="tomato",
        )

        # Mean line
        ax.axhline(
            smoothed.mean(),
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label=f"Mean: {smoothed.mean():.2f}",
        )

        # Range shading
        ax.fill_between(
            x,
            smoothed.min(),
            smoothed.max(),
            alpha=0.08,
            color="tomato",
        )

        # Font sizes
        ax.set_title(
            title,
            fontsize=16,
            fontweight="bold",
            pad=10,
        )

        ax.set_xlabel(
            x_label,
            fontsize=14,
        )

        ax.set_ylabel(
            "Estimated Height (px)",
            fontsize=14,
        )

        ax.tick_params(
            axis="both",
            labelsize=12,
        )

        ax.legend(
            fontsize=12,
            frameon=True,
        )

        ax.grid(
            True,
            linestyle="--",
            linewidth=0.5,
            alpha=0.5,
        )

        fig.tight_layout()

        # Save vector SVG
        if save_svg:
            fig.savefig(
                output_path,
                format="svg",
                bbox_inches="tight",
            )

            print(f"Saved SVG: {output_path}")

        plt.show()

        plt.close(fig)
