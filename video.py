import os
import mediapipe as mp
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.signal as sp_sigal
from collections import deque
from glob import glob
import colorsys 

# signal storage
green_signal_forehead = deque(maxlen=500) # we dont need old frames
green_signal_cheek = deque(maxlen=500)
last_result = None

# after evm
filtered_forehead = deque(maxlen=100) # we dont need old frames
filtered_cheek = deque(maxlen=100)


# MediaPipe setup (thank u papa google)
BaseOptions = mp.tasks.BaseOptions # load model
FaceLandmarker = mp.tasks.vision.FaceLandmarker # create landmarker object
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions # configure landmarker
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode # select mode, e.g VIDEO or LIVE-STREAM


# EVM parameters
# video magnification factor
ALPHA = 50.0
# gaussian pyramid level of which to apply magnification, amplify signal, not noise, level 4 = 1/16 resolution
LEVEL = 4 
# temporal frequency filter params 
FREQ_LOW = 0.4*60 
FREQ_HIGH = 4*60 
# video frame scale factor
SCALE_FACTOR = 1.0


# asynchronous callback
def get_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int): #type: ignore
    global last_result
    last_result = result


# MediaPipe model setup 
def setup_face_landmarker(model_path, running_mode, callback=None):
    base = BaseOptions(model_asset_path=model_path)
    options = FaceLandmarkerOptions(
        base_options=base,
        running_mode=running_mode,
        num_faces=1,
        result_callback=callback if callback else None
    )
    return FaceLandmarker.create_from_options(options)


def get_roi_coords(bb_x1, bb_y1, bb_x2, bb_y2, horizontal_ratio, top_ratio, bottom_ratio, frame_bgr):
    roi_y1 = int(bb_y1 + top_ratio * (bb_y2 - bb_y1))
    roi_y2 = int(bb_y1 + bottom_ratio * (bb_y2 - bb_y1))
    roi_x1 = int(bb_x1 + horizontal_ratio * (bb_x2 - bb_x1))
    roi_x2 = int(bb_x2 - horizontal_ratio * (bb_x2 - bb_x1))
    cv.rectangle(frame_bgr, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
    return roi_x1, roi_y1, roi_x2, roi_y2


def get_avg_green(roi):
    return np.mean(roi[:, :, 1]) # height, width, color


def update_plot(g_forehead, g_cheek, line1, line2, ax):
    line1.set_ydata(g_forehead)
    line1.set_xdata(range(len(g_forehead)))
    line2.set_ydata(g_cheek)
    line2.set_xdata(range(len(g_cheek)))
    # recompute axis limits and update view
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.001)


def process_frame(frame_bgr, landmarks):
    h, w, _ = frame_bgr.shape
    x_vals = [lm.x for lm in landmarks]
    y_vals = [lm.y for lm in landmarks]
    bb_x1 = int(min(x_vals) * w)
    bb_y1 = int(min(y_vals) * h)
    bb_x2 = int(max(x_vals) * w)
    bb_y2 = int(max(y_vals) * h)

    cv.rectangle(frame_bgr, (bb_x1, bb_y1), (bb_x2, bb_y2), (0, 255, 0), 2)

    f_x1, f_y1, f_x2, f_y2 = get_roi_coords(bb_x1, bb_y1, bb_x2, bb_y2, 0.25, 0.00, 0.25, frame_bgr)
    c_x1, c_y1, c_x2, c_y2 = get_roi_coords(bb_x1, bb_y1, bb_x2, bb_y2, 0.15, 0.4, 0.65, frame_bgr)

    roi_forehead = frame_bgr[f_y1:f_y2, f_x1:f_x2]
    roi_cheek = frame_bgr[c_y1:c_y2, c_x1:c_x2]

    avg_green_forehead = get_avg_green(roi_forehead)
    avg_green_cheek = get_avg_green(roi_cheek)

    green_signal_forehead.append(avg_green_forehead)
    green_signal_cheek.append(avg_green_cheek)
    

# coverts a BGR image to float32 YIQ
def bgr2yiq(frame_bgr):
    # get normalized YIQ frame
    rgb = np.float32(cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB))
    yig = colorsys.rgb_to_yiq(rgb)
    return yig


# TODO!  fix params
def fir_bandpass_filter(signal, fs, low, high, running_mode):

    """
    - signal: Input signal (1D array-like)
    - fs: Sampling rate (frames per second)
    - low: Low cutoff frequency (Hz)
    - high: High cutoff frequency (Hz)
    - numtaps: higher num = better freq resolution but slower ( must be < len(available frames ))
    if len(green_signal_forehead) >= numtaps:
        filtered = fir_bandpass_filter(green_signal_forehead, fs, low, high, numtaps)

    - running_mode: LIVE_STREAM / VIDEO (if True, use filtfilt for zero-phase filtering. else, use lfilter. )
    """
    # more accurate for real time processing
    if running_mode == VisionRunningMode.VIDEO:
        taps = sp_sigal.firwin(numtaps=110, cutoff=[low, high], fs=fs, pass_zero=False)
        return sp_sigal.filtfilt(taps, [1.0], signal)
    
    # real - time processing, less accurate
    else:
        taps = sp_sigal.firwin(numtaps=50, cutoff=[low, high], fs=fs, pass_zero=False)
        return sp_sigal.lfilter(taps, [1.0], signal)
    

'''
EVM PROCESS from MIT paper

- get video frames in YIQ colorspace
- obtain single gaussian pyramid level of each frame
- temporal bandpass filter to obtain heart rate between 0.83 to 1.0 Hz
- magnify filtered pyramid levels back
- add magnified pyramid levels back to original frames
- convert back to RGB / BGR color space to display
'''

#def evm(frame_bgr):

def main():

    print("PRESS q to quit -- PRESS spacebar to pause")

    # important variables
    global last_result # last detected frame
    paused = False # boolean flag for pause functionality
    last_frame = None # copy of last detected frame

    # setup interactive plot
    plt.ion()
    _, ax = plt.subplots()
    line1, = ax.plot([], [], label="forehead")
    line2, = ax.plot([], [], label="cheek")
    ax.set_title("green channel signal")
    ax.set_xlabel("frame")
    ax.set_ylabel("filtered green value")
    ax.legend()

    # get model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "face_landmarker.task")

    # select input for VIDEO mode (video file) with user prompt
    choice = input("Select input: [1] light_skin or [2] dark_skin or [3] angus: ").strip()
    if choice == "1":
        video_file = "light_skin.mp4"
    elif choice == "2":
        video_file = "dark_skin.mp4"
    elif choice == "3":
        video_file = "angus.mp4"
    else:
        print("Invalid option. Exiting.")
        exit(1)

    path = os.path.join("video-footage", video_file)
    cam = cv.VideoCapture(path)
    running_mode = VisionRunningMode.VIDEO
    use_callback = False

    landmarker = setup_face_landmarker(model_path, running_mode, get_result if use_callback else None)

    with landmarker:
        while True:
            if not paused:    
                ret, frame_bgr = cam.read()
                # video sampling rate
                #fs = cam.get(cv.CAP_PROP_FPS) # some webcams may give incorrect fps

                # if we hit the end of our video footage (last frame)
                if not ret:
                    print("End of stream reached — freezing displays. Press 'q' to quit.")
                    plt.ioff()
                    plt.show()  # display the final plot
                    # loop to freeze the video window at the last frame.
                    while True:
                        cv.imshow("face landmarker with rois", last_frame)
                        key = cv.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                    break

                # saving current frame so it can be displayed if we hit pause
                last_frame = frame_bgr.copy()

                frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                timestamp = int(time.time() * 1000) if use_callback else int(cam.get(cv.CAP_PROP_POS_MSEC))
                if use_callback:
                    landmarker.detect_async(mp_image, timestamp)
                else:
                    last_result = landmarker.detect_for_video(mp_image, timestamp)

                if last_result and last_result.face_landmarks:
                    # TODO: apply EVM here
                    process_frame(frame_bgr, last_result.face_landmarks[0])
                
                # update dynamic plot
                update_plot(green_signal_forehead, green_signal_cheek, line1, line2, ax)
            else:
                frame_bgr = last_frame

            cv.imshow("face landmarker with rois", frame_bgr)
            
            key = cv.waitKey(1) & 0xFF # wait 1 ms, then extract keycode
            if key == ord('q'):
                break
            # spacebar = pause
            if key == ord(' '):
                paused = not paused

    cam.release()
    cv.destroyAllWindows()


main()

