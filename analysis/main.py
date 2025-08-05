import argparse
import os
import importlib
import importlib.util
from pathlib import Path
import numpy as np

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
    
    truth_path = os.path.join(VIDEOS_DIR, f"{".".join(video_file.split(".")[:-1])}.csv")
    if not os.path.exists(truth_path):
        print(f"❌ Ground Truth Data '{truth_path}' not found in '{VIDEOS_DIR}'")
        return

    base_name = Path(video_file).stem
    degradation_dir = os.path.join(
        RESULTS_DIR, base_name, "degraded", degradation)
    os.makedirs(degradation_dir, exist_ok=True)

    # Store time series estimates for each method per degraded version
    results = {method: {} for method in methods}

    print(f"\n🛠️ Applying degradation: {degradation}")
    for degraded_path, degraded_label in apply_degradation(degradation, video_path):
        print(f"  📽️ Created degraded version: {degraded_label}")

        for method in methods:
            print(f"    📏 Measuring HR using {method} method")
            measurement = apply_measurement(degraded_path, method)
            results[method][degraded_label] = measurement

            # Save individual result
            out_dir = os.path.join(
                RESULTS_DIR, base_name, "measurements", method, degradation)
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(
                out_dir, f"{degraded_label}.npy"), measurement)
    print(f"✅ Saved results\n")

    # Running metrics
    metrics_path = Path("metrics")
    for metric_file in metrics_path.glob("*.py"):
        if metric_file.name.startswith("_"):
            continue

        spec = importlib.util.spec_from_file_location(
            metric_file.stem, metric_file)
        if not spec or not spec.loader:
            continue

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        print(f"📊 Running metric: {metric_file.stem}")
        module.plot(truth_results, results, x_label=degradation,
                    output_dir=metrics_path)

    print(f"\n✅ Saved plots to: {metrics_path}")


if __name__ == "__main__":
    main()
