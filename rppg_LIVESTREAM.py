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
import heartpy as hp


# signal storage
green_signal_forehead = deque(maxlen=500) # we dont need old frames
green_signal_cheek = deque(maxlen=500)

# testing other color signals
red_signal_cheek = deque(maxlen=500) 
blue_signal_cheek = deque(maxlen=500)

last_result = None

# MediaPipe setup (thank u papa google)
BaseOptions = mp.tasks.BaseOptions # load model
FaceLandmarker = mp.tasks.vision.FaceLandmarker # create landmarker object
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions # configure landmarker
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode # select mode, e.g VIDEO or LIVE-STREAM


#  bpm limits
FREQ_LOW = 0.7 # 45 bpm
FREQ_HIGH = 1.66 # 100 bpm



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

    roi_cheek = frame_bgr[c_y1:c_y2, c_x1:c_x2]

    # get average signal of green channel
    green_signal_cheek.append(get_avg(roi_cheek, 1))



    

# coverts a BGR image to float32 YIQ
def bgr2yiq(frame_bgr):
    # get normalized YIQ frame
    rgb = np.float32(cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB))
    yig = colorsys.rgb_to_yiq(rgb)
    return yig




def estimate_bpm(signal, fps, ax=None, label='FFT'):
    """
    using FFT, get largest peak to estimate bpm
    """

    # compute FFT of signal to convert from time to frequency domain  (amplitude and phase)
    fft_vals = np.fft.fft(signal)
    # get frequency values in hz
    freqs = np.fft.fftfreq(len(signal), d=1/fps) # d = sampling period
    magnitudes = np.abs(fft_vals)

    # limit to heart rate range 
    mask = (freqs >= FREQ_LOW) & (freqs <= FREQ_HIGH)
    # if no frequencies found within heart rate range
    if not np.any(mask):
        return None

    freqs_band = freqs[mask]
    mags_band  = magnitudes[mask]

    # dominant in-band frequency
    k = int(np.argmax(mags_band))
    f_peak = float(freqs_band[k])
    bpm = f_peak * 60.0

    # plotting
    if ax is not None:
        ax.cla()
        ax.plot(freqs, magnitudes, lw=1, label='|FFT|')
        band_mags = np.where(mask, magnitudes, np.nan)
        ax.plot(freqs, band_mags, lw=2, label='HR band')
        ax.plot([f_peak], [mags_band[k]], 'o', ms=8)
        ax.set_title(f'{label} spectrum (peak → {bpm:.1f} bpm)')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.legend()
        ax.figure.canvas.draw()
        ax.figure.canvas.flush_events()


    return bpm




def estimate_bpm_welch(signal, fps,  ax=None, label='Welch PSD'):
    """
    Estimate BPM using Welch's PSD.
    """

    x = np.asarray(signal, dtype=np.float32)

    # detrend to suppress DC and slow drift
    x = x - np.nanmean(x)

    # Welch params: Responsiveness vs resolution: 
    # shorter segments react faster but quantize coarsely, longer segments smoother but lag
    # breaks the signal into overlapping windows of ~9 seconds each
    window_seconds = 9
    nperseg = int(min(len(x), fps * window_seconds)) # window length
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

    # dominant in-band frequency
    k = int(np.argmax(p_band))
    f_peak = f_band[k]
    bpm = float(f_peak * 60.0)

    # visualisation
    if ax is not None:
        ax.cla()

        # full spectrum of psd
        ax.plot(freqs, psd, lw=1, label='Welch PSD')

        # shade HR band
        ax.fill_between(freqs, psd, where=band_mask, step='mid', alpha=0.25, label='HR band')

        # mark dominant in-band frequency
        ax.axvline(f_peak, linestyle='--', linewidth=1)
        ax.plot([f_peak], [p_band[k]], 'o', ms=7)

        # annotate
        ax.set_title(f'{label}: peak {f_peak:.3f} Hz  →  {bpm:.1f} BPM')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power spectral density')
        ax.grid(True, alpha=0.3)

        # x-limits: 0–3 Hz (~0–180 BPM)
        ax.set_xlim(0, max(3.0, FREQ_HIGH + 0.2))

        ax.legend(loc='upper right')
        ax.figure.canvas.draw()
        ax.figure.canvas.flush_events()

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

    # green signal plot
    _, ax = plt.subplots()
    ax.set_title("Heart Rate bpm")
    ax.set_xlabel("frame")
    ax.set_ylabel("signal value")
    line_gc, = ax.plot([], [], color="green")

    """
    bpm modes: green/red/blue
    bpm modes: heartpy/welch/fft
    """ 

    ax.legend()

    # get model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "face_landmarker.task")


    # 0 = ahila iphone - need to plug in iphone
    # 1 = webcam
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("error: could not open webcam.")
        exit()
    running_mode = VisionRunningMode.LIVE_STREAM
    use_callback = True

    landmarker = setup_face_landmarker(model_path, running_mode, get_result if use_callback else None)

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

 
                last_frame = frame_bgr.copy()

                frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                timestamp = int(time.time() * 1000) if use_callback else int(cam.get(cv.CAP_PROP_POS_MSEC))

                if use_callback:
                    landmarker.detect_async(mp_image, timestamp)

                if last_result and last_result.face_landmarks:
                    process_frame(frame_bgr, last_result.face_landmarks[0])
             
                
                # update dynamic plot (unecessary to make dynamic for non-live footage, maybe change this later)
                signals = [green_signal_cheek]
                lines = [line_gc]
                update_plot(signals, lines, ax)


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
