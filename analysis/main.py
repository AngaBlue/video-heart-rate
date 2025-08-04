import argparse
import os
import importlib
from pathlib import Path
import numpy as np
from utils.video_io import get_video_files
# from utils.plotting import generate_hr_vs_degradation_plot

VIDEOS_DIR = "videos"
RESULTS_DIR = "results"
DEGRADATION_DIR = "degradation"
MEASUREMENT_DIR = "measurement"


def dynamic_import(module_path: str):
    return importlib.import_module(module_path)


def apply_degradation(degradation_type, video_path):
    """Yield tuples of (degraded_video_path, label)"""
    module = dynamic_import(f"{DEGRADATION_DIR}.{degradation_type}")
    return module.apply(video_path)  # Expected to be a generator


def apply_measurement(video_path, method_name):
    module = dynamic_import(f"{MEASUREMENT_DIR}.{method_name}")
    return module.measure(video_path)


def main():
    parser = argparse.ArgumentParser(
        description="HR Estimation under Degradation")
    parser.add_argument('--video', type=str, required=True,
                        help='Input video filename')
    parser.add_argument('--degradation', type=str,
                        required=True, help='Degradation technique to use')
    parser.add_argument('--methods', nargs='+', required=True,
                        help='Measurement methods to apply')
    args = parser.parse_args()

    video_file = args.video
    degradation = args.degradation
    methods = args.methods

    video_path = os.path.join(VIDEOS_DIR, video_file)
    if not os.path.exists(video_path):
        print(f"❌ Video '{video_file}' not found in '{VIDEOS_DIR}'")
        return

    base_name = Path(video_file).stem
    degradation_dir = os.path.join(
        RESULTS_DIR, base_name, "degraded", degradation)
    os.makedirs(degradation_dir, exist_ok=True)

    # Store average HR for each method per degraded version
    hr_results = {method: {} for method in methods}

    print(f"\n🛠️ Applying degradation: {degradation}")
    for degraded_path, label in apply_degradation(degradation, video_path):
        print(f" -> Degraded version: {label}")

        for method in methods:
            print(f"    -> Measuring HR using {method}")
            hr_signal = apply_measurement(degraded_path, method)
            avg_hr = np.mean(hr_signal)

            hr_results[method][label] = avg_hr

            # Save individual result
            out_dir = os.path.join(
                RESULTS_DIR, base_name, "hr_outputs", method, degradation)
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, f"{label}.npy"), hr_signal)

    # # ✅ Plotting
    # print("\n📈 Generating HR vs Degradation plot...")
    # plot_dir = os.path.join(RESULTS_DIR, base_name, "graphs")
    # os.makedirs(plot_dir, exist_ok=True)
    # plot_path = os.path.join(plot_dir, f"avg_hr_vs_{degradation}.png")

    # generate_hr_vs_degradation_plot(hr_results, plot_path, x_label=degradation)
    # print(f"✅ Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
