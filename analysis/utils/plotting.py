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

    for method, results in hr_results.items():
        # Plot skipping missing values
        plt.plot(list(results.keys()), list(results.values()), marker='o', label=method)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

