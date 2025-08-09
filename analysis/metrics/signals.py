from typing import Dict, Set
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.video_io import interpolate_hr_to_frames

def plot(
    truth: pd.DataFrame,
    results: Dict[str, Dict[str, np.ndarray]],
    output_dir: str,
    x_label: str,
) -> None:
    """
    Plot all BPM signals on a single figure:
      - One line per (method, degradation)
      - Truth drawn once as a step curve

    Args:
        truth: DataFrame with ['timestamp','heart_rate'] (seconds).
        results: {method: {degradation: measured (N,2)}} with measured[:,0]=t(s), measured[:,1]=BPM.
        output_dir: Directory to save the figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))

    # Plot each method/degradation series (preserve insertion order; no sorting)
    truth_interpolated = None
    for method, by_deg in results.items():
        for degradation, measured in by_deg.items():
            if not isinstance(truth_interpolated, np.ndarray):
                truth_interpolated = interpolate_hr_to_frames(truth, measured)

            ax.plot(measured[:, 0], measured[:, 1], linewidth=1.25, label=f"{method} - {degradation}")

    # Plot truth once across full duration
    if isinstance(truth_interpolated, np.ndarray):
        ax.plot(truth_interpolated[:, 0], truth_interpolated[:, 1], linewidth=1.6, label='Truth')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("BPM")
    ax.set_title("BPM over Time")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()

    save_path = os.path.join(output_dir, f"signals_{x_label}.png")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
