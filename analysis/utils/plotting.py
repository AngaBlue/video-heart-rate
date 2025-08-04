import matplotlib.pyplot as plt
from typing import Dict


def generate_hr_vs_degradation_plot(hr_results: Dict[str, Dict[str, float]],
                                    save_path: str,
                                    x_label: str = "Degradation Level") -> None:
    """
    Plots average heart rate vs degradation level for multiple measurement methods.

    Args:
        hr_results: Dict mapping method name -> {degradation_label -> avg_hr}
        save_path: Path to save the plot PNG
        x_label: Label for the X-axis (e.g., "Spatial Resolution")
    """
    plt.figure(figsize=(10, 6))
    plt.title(f"Average Heart Rate vs {x_label.replace('_', ' ').title()}")
    plt.xlabel(x_label.replace('_', ' ').title())
    plt.ylabel("Average Heart Rate (BPM)")

    # Get sorted unique labels from all methods
    all_labels = set()
    for method_dict in hr_results.values():
        all_labels.update(method_dict.keys())
    sorted_labels = sorted(all_labels, key=lambda x: _sort_key(x))

    for method, results in hr_results.items():
        # Extract average HR values in sorted order; use None or np.nan if missing
        y_vals = [results.get(label, None) for label in sorted_labels]

        # Plot skipping missing values
        plt.plot(sorted_labels, y_vals, marker='o', label=method)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _sort_key(label: str):
    """
    Helper to sort degradation labels logically.
    Examples:
        "original" -> 0
        "853x480" -> 480 (height)
        "640x360" -> 360 (height)
        "25fps" -> 25 (fps)
        "8-bit" -> 8 (bits)
    """
    if label == "original":
        return 0
    # Try to parse common patterns:
    if "x" in label:
        # e.g. "853x480" -> sort by height (second number)
        try:
            parts = label.split("x")
            return int(parts[1])
        except Exception:
            return label
    elif "fps" in label:
        try:
            return int(label.replace("fps", ""))
        except Exception:
            return label
    elif "bit" in label:
        try:
            return int(label.split("-")[0])
        except Exception:
            return label
    else:
        return label
