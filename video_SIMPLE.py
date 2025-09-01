import os
import mediapipe as mp
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.signal as sp
from collections import deque
from glob import glob
import colorsys 

# signal storage
green_signal_forehead = deque(maxlen=500) # we dont need old frames
green_signal_cheek = deque(maxlen=500)

# testing other color signals
red_signal_cheek = deque(maxlen=500) 
blue_signal_cheek = deque(maxlen=500)

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



# temporal frequency filter params 
FREQ_LOW = 0.7 # 45 bpm
FREQ_HIGH = 1.66 # 100 bpm



# MediaPipe model setup 
def setup_face_landmarker(model_path, running_mode):
    base = BaseOptions(model_asset_path=model_path)
    options = FaceLandmarkerOptions(
        base_options=base,
        running_mode=running_mode,
        num_faces=1
    )
    return FaceLandmarker.create_from_options(options)


def get_roi_coords(bb_x1, bb_y1, bb_x2, bb_y2, horizontal_ratio, top_ratio, bottom_ratio, frame_bgr):
    roi_y1 = int(bb_y1 + top_ratio * (bb_y2 - bb_y1))
    roi_y2 = int(bb_y1 + bottom_ratio * (bb_y2 - bb_y1))
    roi_x1 = int(bb_x1 + horizontal_ratio * (bb_x2 - bb_x1))
    roi_x2 = int(bb_x2 - horizontal_ratio * (bb_x2 - bb_x1))
    cv.rectangle(frame_bgr, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
    return roi_x1, roi_y1, roi_x2, roi_y2



def get_avg(roi, color):
    """
    red = 0
    green = 1
    blue = 2
    """
    return np.mean(roi[:, :, color]) # height, width, color



def update_plot(signals, line, ax):

    if len(signals) != len(line):
        print("incorrect dimensions")
        return
                    
    for i in range(len(signals)):
        line[i].set_ydata(signals[i])
        line[i].set_xdata(range(len(signals[i])))

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

    # get average signal of green channel
    green_signal_forehead.append(get_avg(roi_forehead, 1))  # height, width, color
    green_signal_cheek.append(get_avg(roi_cheek, 1))

    # testing, compare with red and blue signals
    red_signal_cheek.append(get_avg(roi_cheek, 0))
    blue_signal_cheek.append(get_avg(roi_cheek, 2))


    

# coverts a BGR image to float32 YIQ
def bgr2yiq(frame_bgr):
    # get normalized YIQ frame
    rgb = np.float32(cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB))
    yig = colorsys.rgb_to_yiq(rgb)
    return yig




#get dominant frequency

def estimate_bpm(signal, fps):

    # compute FFT of signal to convert from time to frequency domain  (amplitude and phase)
    fft_vals = np.fft.fft(signal)
    # get frequency values in hz
    freqs = np.fft.fftfreq(len(signal), d=1/fps) # d = sampling period

    # only keep positive frequencies
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    fft_vals = np.abs(fft_vals[pos_mask]) #

    # limit to heart rate range 
    mask = (freqs >= FREQ_LOW) & (freqs <= FREQ_HIGH)

    # if no frequencies found within heart rate range
    if not np.any(mask):
        return None

    freqs_in_band = freqs[mask]
    magnitudes = fft_vals[mask]

    # get frequency with max power (largest magnitude)
    dominant_freq = freqs_in_band[np.argmax(magnitudes)]

    # convert Hz to BPM
    bpm = dominant_freq * 60.0
    return bpm


def estimate_bpm_welch(signal, fps):
    """
    Estimate BPM using Welch's PSD.
    """

    x = np.asarray(signal, dtype=np.float32)

    # detrend to suppress DC and slow drift
    x = x - np.nanmean(x)
    

    # Choose Welch params: ~4–8 s windows if available, with 50% overlap
    target_seconds = 6.0
    # split signal into shorter segments between 128 and 2048, never exceed length
    # segment length affects frequency resolution and variance
    # larger nperseg = finer BPM resolution, but noisier PSD

    # check surrounding bpm, are we quantising ??
    nperseg = int(min(len(x), max(128, min(int(fps * target_seconds), 2048))))
    noverlap = nperseg // 2

    # Welch PSD: how the power of a signal is distributed across frequencies.
    freqs, psd = sp.welch(
        x, fs=fps, window='hann', nperseg=nperseg, noverlap=noverlap,
        detrend='constant', scaling='density', average='mean' # average="median" if more noisy
    )
    
    # limit to heart-rate band
    band_mask = (freqs >= FREQ_LOW) & (freqs <= FREQ_HIGH)
    if not np.any(band_mask):
        return None

    f_band = freqs[band_mask]
    p_band = psd[band_mask]
    if p_band.size == 0 or np.all(p_band <= 0):
        return None

    # Peak in-band
    k = int(np.argmax(p_band))
    f_peak = f_band[k]

    # Convert to BPM
    bpm = float(f_peak * 60.0)
    return bpm










# TRY USING MOVING AVERAGE FILTER
def bandpass_butterworth(signal, fps, freq_lo, freq_high, order):
    """
    Apply Butterworth bandpass filter.
    Input signal shape: [T,] or [T, N]

    more stable bc breaks down a high-order filter into cascaded second-order sections (SOS),
    """
    nyquist = 0.5 * fps
    low = freq_lo / nyquist
    high = freq_high / nyquist
    
    sos_butter = sp.butter(order, [low, high], btype='band', output='sos')
    filtered = sp.sosfiltfilt(sos_butter, signal, axis=0)
    
    return filtered



def main():

    print("PRESS q to quit -- PRESS spacebar to pause")

    # important variables
    global last_result # last detected frame
    video_frames = []  # will store raw frames (BGR)
    paused = False # boolean flag for pause functionality
    last_frame = None # copy of last detected frame

    # setup interactive plot
    plt.ion()
    _, ax = plt.subplots()
    #line_gf, = ax.plot([], [], label="forehead green", color="dark green")
    line_gc, = ax.plot([], [], color="green")
    #line_r, = ax.plot([], [], label="cheek red", color="red")
    #line_b, = ax.plot([], [], label="cheek blue", color="blue")

    ax.set_title("Heart Rate bpm")
    ax.set_xlabel("frame")
    ax.set_ylabel("signal value")

    bpm_green_text = ax.text(0.95, 1, '', transform=ax.transAxes, ha='right', va='top')
    #bpm_red_text = ax.text(0.95, 0.85, '', transform=ax.transAxes, ha='right', va='top')
    #bpm_blue_text = ax.text(0.95, 0.80, '', transform=ax.transAxes, ha='right', va='top')

    ax.legend()

    # get model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(script_dir, 'videos')
    model_path = os.path.join(script_dir, "face_landmarker.task")

    # select input for VIDEO mode (video file) with user prompt
    print("Select input video file:")
    files = list(os.listdir(video_dir))
    for i, path in enumerate(files):
        if path != ".gitignore":
            print(f"[{i + 1}] {path}")
    choice = int(input().strip()) - 1

    if choice < 0 or choice >= len(files):
        print("Invalid choice, exiting...")
        exit(1)
    
    path = os.path.join(video_dir, files[choice])
    cam = cv.VideoCapture(path)
    running_mode = VisionRunningMode.VIDEO
    use_callback = False

    landmarker = setup_face_landmarker(model_path, running_mode)

    with landmarker:
        while True:
            if not paused:    
                ret, frame_bgr = cam.read()

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

                # video sampling rate
                fps = cam.get(cv.CAP_PROP_FPS)
                last_frame = frame_bgr.copy()
                video_frames.append(last_frame)

                frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                timestamp = int(time.time() * 1000) if use_callback else int(cam.get(cv.CAP_PROP_POS_MSEC))
                if use_callback:
                    landmarker.detect_async(mp_image, timestamp)
                else:
                    last_result = landmarker.detect_for_video(mp_image, timestamp)

                if last_result and last_result.face_landmarks:
                    process_frame(frame_bgr, last_result.face_landmarks[0])
                
                # update dynamic plot

                signals = [green_signal_cheek]
                lines = [line_gc]

                update_plot(signals, lines, ax)
                


                # apply bandpass filter to cheek signal, using 2nd order butterworth

                if len(green_signal_cheek) > 100:  # ensure enough signal length
                 
                    filtered_green = bandpass_butterworth(np.array(green_signal_cheek, dtype=np.float32), fps, FREQ_LOW, FREQ_HIGH, order=2)
                    #filtered_red = bandpass_butterworth(np.array(red_signal_cheek, dtype=np.float32), fps, FREQ_LOW, FREQ_HIGH, order=2)
                    #filtered_blue = bandpass_butterworth(np.array(blue_signal_cheek, dtype=np.float32), fps, FREQ_LOW, FREQ_HIGH, order=2)

                    bpm_green = estimate_bpm_welch(filtered_green, fps)
                    #bpm_red = estimate_bpm(filtered_red, fps)
                    #bpm_blue = estimate_bpm(filtered_blue, fps)

                    
                    if bpm_green is not None:
                        print(f"Estimated BPM GREEN: {bpm_green:.2f}")
                        #print(f"Estimated BPM RED : {bpm_red:.2f}")
                        #print(f"Estimated BPM BLUE : {bpm_blue:.2f}")
                        bpm_green_text.set_text("")
                        bpm_green_text.set_text(f"BPM GREEN: {bpm_green:.1f}")

                        '''bpm_red_text.set_text("")
                        bpm_red_text.set_text(f"BPM RED: {bpm_red:.1f}")

                        bpm_blue_text.set_text("")
                        bpm_blue_text.set_text(f"BPM BLUE: {bpm_blue:.1f}")'''

  

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
