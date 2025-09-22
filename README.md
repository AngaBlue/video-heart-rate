# 📊 Video Heart Rate

Remote photoplethysmography (rPPG) uses cameras to non-invasively monitor vital signals like heart rate, offering contactless monitoring with potential for remote healthcare applications.

## 🛠️ Setup
At the root of the project (`video-heart-rate`) run the following to set-up with Python virtual environments:

```bash
python3 -m venv .venv
```

Then activate the virtual environmental and install dependencies with:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Development
To speed up processing, the video can be degraded significantly to reduce unnecessary information using the following command.

```bash
ffmpeg -y -i .\videos\angus.mp4 -c:v libx264 -vf scale=144:-2 -r 5 -an -pix_fmt yuv420p -crf 20 .\videos\angus-degraded.mp4
```

This will:
 - Re-encode the video with H.264
 - Reduce the framerate to 5 FPS.
 - Reduce the resolution to 144p.
 - Set the constant rate factor (CRF) to 20.

## 📝 Notes
The pause functionality only works if the CV window is active.

The following resources have been used to create this project:

 - https://people.csail.mit.edu/mrub/evm/ - Website
 - https://people.csail.mit.edu/mrub/papers/vidmag.pdf - EVM, video footage
 - https://github.com/itberrios/CV_projects/tree/main/color_mag - Python implementation of EVM
 - https://www.sciencedirect.com/science/article/pii/S0957417423006371?via%3Dihub#s0025 - Overall structure 


NOTE: if you want to use recorded iphone footage, iPhone: Settings → Camera → Record Video → disable HDR., else footage is super pale
