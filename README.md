# video-heart-rate

Remote photoplethysmography (rPPG) uses cameras to non-invasively monitor vital signals like heart rate, offering contactless monitoring with potential for remote healthcare applications.


SETUP (mac-os)

1. source .venv/bin/activate (if u already have ur .venv)
1. python3 -m venv .venv (if u need to create ur .venv)

2. pip install -r requirements.txt


Notes:
pause functionality only works if u press on CV window

I am referencing the following papers / repositories:

https://people.csail.mit.edu/mrub/papers/vidmag.pdf - EVM, video footage
https://github.com/itberrios/CV_projects/tree/main/color_mag - python implementation of EVM
https://www.sciencedirect.com/science/article/pii/S0957417423006371?via%3Dihub#s0025 - overall structure 


ISSUES:

shape/size issues with gausian pyramid when there are an odd number of frames
