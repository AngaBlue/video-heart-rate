from typing import Dict
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from utils.video_io import interpolate_hr_to_frames


def compute_mae_per_method_degradation(
    truth: pd.DataFrame,
    results: Dict[str, Dict[str, np.ndarray]]
) -> pd.DataFrame:
    """
    Compute mean absolute error (MAE) for each (method, degradation).

    Args:
        truth: DataFrame with columns ['timestamp', 'heart_rate'] (seconds).
        results: Dict like {method: {degradation: measured}}
                 where each measured is an (N, 2) float array:
                 column 0 = timestamps (s), column 1 = predicted HR.

    Returns:
        DataFrame with columns ['method', 'degradation', 'mae'].
    """
    rows = []

    for method, by_deg in results.items():
        for degradation, measured in by_deg.items():
            aligned_truth = interpolate_hr_to_frames(truth, measured)
            truth_hr = aligned_truth[:, 1].astype(float)
            pred_hr  = measured[:, 1].astype(float)
            mae = float(np.mean(np.abs(pred_hr - truth_hr)))
            rows.append({"method": method, "degradation": degradation, "mae": mae})

    return pd.DataFrame(rows, columns=["method", "degradation", "mae"])


def plot(
    truth: pd.DataFrame,
    results: Dict[str, Dict[str, np.ndarray]],
    x_label: str,
    output_dir: str,
) -> None:
    """
    Create a simple plot of MAE vs degradation for each method.

    Args:
        truth: Ground-truth DataFrame ['timestamp','heart_rate'].
        results: {method: {degradation: measured (N,2)}}
        x_label: Label for the x-axis (e.g., "Downscale factor", "Bit depth", etc.)
        save_path: Optional path to save the figure (PNG). If None, shows the plot.
    """
    mae_df = compute_mae_per_method_degradation(truth, results)

    # Keep degradation order as it appears per method; otherwise treat as categorical
    # You can enforce a custom order by converting 'degradation' to a Categorical beforehand.
    fig, ax = plt.subplots(figsize=(9, 5))

    for method, sub in mae_df.groupby("method", sort=False):
        # Sort by degradation label string for reproducibility; adjust to suit your labels
        sub = sub.sort_values("degradation")
        ax.plot(sub["degradation"], sub["mae"], marker="o", label=method)

    ax.set_xlabel(x_label)
    ax.set_ylabel("MAE (|predicted HR - truth HR|)")
    ax.set_title(f"Mean Absolute Error vs {x_label}")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Method")
    fig.tight_layout()

    save_path = os.path.join(output_dir, f'mae_vs_{x_label}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)

