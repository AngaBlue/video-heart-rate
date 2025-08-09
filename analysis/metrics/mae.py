from typing import Dict
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.video_io import interpolate_hr_to_frames


def plot(
    truth: pd.DataFrame,
    results: Dict[str, Dict[str, np.ndarray]],
    x_label: str,
    output_dir: str,
) -> None:
    """
    Plot MAE vs degradation for each method, computing MAE inline and preserving
    the degradation order as in the results dict (no sorting).

    Args:
        truth: DataFrame with ['timestamp','heart_rate'] (seconds).
        results: {method: {degradation: measured (N,2)}} with measured[:,0]=t(s), measured[:,1]=pred HR (BPM).
        x_label: X-axis label (e.g., "Spatial Resolution", "Colour Noise", etc.).
        output_dir: Directory to save figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for method, by_deg in results.items():
        labels = []
        maes = []

        for degradation, measured in by_deg.items():
            aligned_truth = interpolate_hr_to_frames(truth, measured)
            truth_hr = aligned_truth[:, 1].astype(float)
            pred_hr = measured[:, 1].astype(float)

            mae = float(np.mean(np.abs(pred_hr - truth_hr)))
            labels.append(degradation)
            maes.append(mae)

        if labels:
            ax.plot(labels, maes, marker="o", label=method)

    ax.set_xlabel(x_label)
    ax.set_ylabel("MAE (|predicted HR - truth HR|)")
    ax.set_title(f"Mean Absolute Error vs {x_label}")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Method")
    fig.tight_layout()

    save_path = os.path.join(output_dir, f"mae_vs_{x_label}.png")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
