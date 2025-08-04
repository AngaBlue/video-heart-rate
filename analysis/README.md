# 📊 Video Heart Rate Analysis System

A modular Python system for investigating heart rate estimation techniques on pre-recorded video, with support for applying video degradations and generating comparison plots.

---

## 🚀 Quick Start

### 1. 📁 Place a Video

Put your input video(s) in the `videos/` folder. For example:

```
videos/sample1.mp4
```

_**Note**: videos added to this folder are git ignored, meaning they won't be committed to avoid filling the repository with large files._

### 2. ▶️ Run the CLI

Use the `main.py` script to apply one degradation technique and evaluate multiple HR measurement techniques:

```bash
python main.py --video sample1.mp4 \
               --degradation spatial_resolution \
               --methods green_channel
```

This will:
 - Apply multiple levels of spatial resolution degradation.
 - Measure average heart rate from each degraded video using selected methods.
 - Save results in `results/` and generate a comparison plot.

## 📁 Project Structure
```
video_hr_analysis/
├── main.py                     # CLI orchestration
├── videos/                     # Input videos
├── results/                    # Output videos, HR data, plots
│
├── degradation/                # Video degradation modules│
├── measurement/                # Heart rate estimation techniques│
├── utils/
│   ├── video_io.py             # Read/write video
│   ├── plotting.py             # Generate HR vs degradation plots
│   └── metrics.py              # (optional) Metrics like MAE, RMSE
```

## 🧩 Adding a New Degradation Method
Each degradation module must:
 - Be placed in the degradation/ folder.
 - Export a function `apply(input_path: str) -> Generator[Tuple[str, str]]`.
 - Yield (`degraded_video_path`, `label`) for each output version.

For example, a colour quantisation/bit-depth degradation would be written as follows:
```python
def apply(input_path: str):
    for bits in [8, 6, 4]:
        # create quantized video...
        yield output_path, f"{bits}-bit"
```

## 🧠 Adding a New HR Measurement Technique
Each method must:
 - Be placed in the `measurement/` folder.
 - Export a function `measure(video_path: str) -> np.ndarray` that returns a time-series signal (e.g., HR over time).

For example, a green channel over face ROI would be written as follows:
```python
def measure(video_path: str) -> np.ndarray:
    # extract green channel, average over face ROI, etc.
    return np.array([...])
```

## 📈 Output
The system saves:
 - Degraded videos: `results/<video>/degraded/<technique>/<label>.mp4`
 - HR signals: `results/<video>/hr_outputs/<method>/<technique>/<label>.npy`
 - Graphs: `results/<video>/graphs/avg_hr_vs_<technique>.png`
