import os
import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from collections import deque
import sys

#import sys
   #sys.path.insert(0,'/path/to/mod_directory')



# signal storage (for ROI extraction demo)
green_signal_forehead = deque(maxlen=500)  # we don't need old frames
green_signal_cheek = deque(maxlen=500)
last_result = None

# EVM parameters
ALPHA = 50.0          # magnification factor
LEVEL = 4             # number of downscaling steps in the Gaussian pyramid
FREQ_LOW = 0.4        # low frequency cutoff in Hz 
FREQ_HIGH = 4         # high frequency cutoff in Hz

###########################
#   Color Magnification   # CITATION: https://github.com/itberrios/CV_projects/tree/main/color_mag
###########################


def rgb2yiq(rgb):
    """Converts an RGB image to YIQ using FCC NTSC format.
        (using built in colorsys library is way too slow)
    """
    y = rgb @ np.array([[0.30], [0.59], [0.11]])
    rby = rgb[:, :, (0, 2)] - y
    i = np.sum(rby * np.array([[[0.74, -0.27]]]), axis=-1)
    q = np.sum(rby * np.array([[[0.48, 0.41]]]), axis=-1)
    yiq = np.dstack((y.squeeze(), i, q))
    return yiq

def bgr2yiq(bgr):
    """Converts a BGR image to float32 YIQ."""
    rgb = np.float32(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    yiq = rgb2yiq(rgb / 255)  # normalize to [0,1]
    return yiq

def yiq2rgb(yiq):
    """Converts a YIQ image to RGB.
    """
    r = yiq @ np.array([1.0, 0.9468822170900693, 0.6235565819861433])
    g = yiq @ np.array([1.0, -0.27478764629897834, -0.6356910791873801])
    b = yiq @ np.array([1.0, -1.1085450346420322, 1.7090069284064666])
    rgb = np.clip(np.dstack((r, g, b)), 0, 1)
    return rgb

# inverse colorspace conversion: YIQ -> RGB, then normalize to uint8.
inv_colorspace = lambda x: cv2.normalize(yiq2rgb(x), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)


def compute_downsampled_size(shape, level):
    rows, cols, _ = shape
    for _ in range(level):
        rows = (rows + 1) // 2
        cols = (cols + 1) // 2
    return rows, cols



def gaussian_pyramid(image, level):
    """Obtains a single band from a Gaussian Pyramid decomposition.
    Inputs: 
        image - single RGB image (float or uint8) with shape (rows, cols, 3)
        level - number of pyramid levels
    Outputs:
        pyramid - Array of shape (3, rows//(2**level), cols//(2**level))
                  containing the final Gaussian pyramid level (for each color channel)
    """

    rows, cols = compute_downsampled_size(image.shape, level)
    colors = image.shape[2]
    pyramid = np.zeros((colors, rows, cols))
    
    for i in range(level):
        image = cv2.pyrDown(image)
    for c in range(colors):
        pyramid[c, :, :] = image[:, :, c]
    return pyramid


def mag_colors(rgb_frames, fs, freq_lo, freq_hi, level, alpha):
    # use the compute_downsampled_size function to get the exact dimensions.
    rows, cols = compute_downsampled_size(rgb_frames[0].shape, level)
    colors = rgb_frames[0].shape[2]
    num_frames = len(rgb_frames)
    pyramid_stack = np.zeros((num_frames, colors, rows, cols))
    
    # convert frames to YIQ colorspace (normalize by 255)
    frames = [rgb2yiq(frame / 255.0) for frame in rgb_frames]
    
    # temporal filtering: design ideal bandpass FIR filter.
    bandpass = signal.firwin(numtaps=num_frames,
                             cutoff=(freq_lo, freq_hi),
                             fs=fs,
                             pass_zero=False)
    transfer_function = np.fft.fft(np.fft.ifftshift(bandpass))
    transfer_function = transfer_function[:, None, None, None].astype(np.complex64)
    
    # build Gaussian pyramid for each frame.
    for i, frame in enumerate(frames):
        pyramid = gaussian_pyramid(frame, level)
        pyramid_stack[i, :, :, :] = pyramid  # Now dimensions match
    
    # apply temporal filtering in the frequency domain.
    pyr_stack_fft = np.fft.fft(pyramid_stack, axis=0).astype(np.complex64)
    filtered_pyramid = np.fft.ifft(pyr_stack_fft * transfer_function, axis=0).real
    
    # magnify the filtered signal.
    magnified_pyramid = filtered_pyramid * alpha
    
    # reconstruct video: upsample each pyramid level and add back to original signal.
    magnified = []
    for i in range(num_frames):
        y_chan = frames[i][:, :, 0]
        i_chan = frames[i][:, :, 1]
        q_chan = frames[i][:, :, 2]
    
        fy_chan = cv2.resize(magnified_pyramid[i, 0, :, :], (rgb_frames[0].shape[1], rgb_frames[0].shape[0]))
        fi_chan = cv2.resize(magnified_pyramid[i, 1, :, :], (rgb_frames[0].shape[1], rgb_frames[0].shape[0]))
        fq_chan = cv2.resize(magnified_pyramid[i, 2, :, :], (rgb_frames[0].shape[1], rgb_frames[0].shape[0]))
    
        mag_frame = np.dstack((
            y_chan + fy_chan,
            i_chan + fi_chan,
            q_chan + fq_chan,
        ))
        mag_frame = inv_colorspace(mag_frame)
        magnified.append(mag_frame)
    return magnified


###############################
#   Face Detection and ROI    #
###############################

# global variables to store ROI data and final detection parameters.
raw_roi_signal = []       # raw ROI (forehead) green channel values (one per frame)
final_bb = None           # final detected face bounding box (x1, y1, x2, y2)
final_roi_coords = None   # final ROI coordinates (x1, y1, x2, y2)

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def get_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):  # type: ignore
    global last_result
    last_result = result

def setup_face_landmarker(model_path):
    base = BaseOptions(model_asset_path=model_path)
    options = FaceLandmarkerOptions(
        base_options=base,
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1
    )
    return FaceLandmarker.create_from_options(options)

def get_roi_coords(bb_x1, bb_y1, bb_x2, bb_y2, horizontal_ratio, top_ratio, bottom_ratio):
    """Calculates the ROI relative to the face bounding box."""
    roi_y1 = int(bb_y1 + top_ratio * (bb_y2 - bb_y1))
    roi_y2 = int(bb_y1 + bottom_ratio * (bb_y2 - bb_y1))
    roi_x1 = int(bb_x1 + horizontal_ratio * (bb_x2 - bb_x1))
    roi_x2 = int(bb_x2 - horizontal_ratio * (bb_x2 - bb_x1))
    return roi_x1, roi_y1, roi_x2, roi_y2

def get_avg_green(roi):
    """Returns the average green channel value (assumes ROI in BGR format)."""
    return np.mean(roi[:, :, 1])

def process_frame(frame_bgr, landmarks):
    """
    For a given frame and detected face landmarks, compute the face bounding box,
    define a forehead ROI, and record its average green value.
    """
    global final_bb, final_roi_coords, raw_roi_signal
    h, w, _ = frame_bgr.shape
    x_vals = [lm.x for lm in landmarks]
    y_vals = [lm.y for lm in landmarks]
    bb_x1 = int(min(x_vals) * w)
    bb_y1 = int(min(y_vals) * h)
    bb_x2 = int(max(x_vals) * w)
    bb_y2 = int(max(y_vals) * h)
    final_bb = (bb_x1, bb_y1, bb_x2, bb_y2)
    
    roi_coords = get_roi_coords(bb_x1, bb_y1, bb_x2, bb_y2,
                                horizontal_ratio=0.25,
                                top_ratio=0.0,
                                bottom_ratio=0.25)
    final_roi_coords = roi_coords
    roi_forehead = frame_bgr[roi_coords[1]:roi_coords[3], roi_coords[0]:roi_coords[2]]
    avg_green = get_avg_green(roi_forehead)
    raw_roi_signal.append(avg_green)

def draw_roi_on_frame(frame, bb, roi_coords):
    """
    Draws the face bounding box (green) and ROI (blue) on the provided frame.
    Frame is expected to be in BGR format.
    """
    out_frame = frame.copy()
    if bb is not None:
        cv2.rectangle(out_frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
    if roi_coords is not None:
        cv2.rectangle(out_frame, (roi_coords[0], roi_coords[1]),
                      (roi_coords[2], roi_coords[3]), (255, 0, 0), 2)
    return out_frame

def extract_roi_signal(frames, roi_coords):
    """
    Given a list of frames and ROI coordinates (x1, y1, x2, y2),
    returns the time series (one value per frame) of the average green channel within the ROI.
    Frames are assumed to be in BGR.
    """
    signal_values = []
    x1, y1, x2, y2 = roi_coords
    for frame in frames:
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            signal_values.append(np.mean(roi[:, :, 1]))
        else:
            signal_values.append(0)
    return signal_values



# TODO
def calculate_bpm(signal, fps):
   pass



###############################
#          MAIN CODE         #
###############################

def main():
    global last_result
    video_frames = []  # will store raw frames (BGR)
    
    # Get model path (assumes face_landmarker.task is in the same folder as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "face_landmarker.task")
    
    # choose video input
    choice = input("Select input: [1] light_skin or [2] dark_skin: ").strip()
    if choice == "1":
        video_file = "light_skin.mp4"
    elif choice == "2":
        video_file = "dark_skin.mp4"
    else:
        print("Invalid option. Exiting.")
        sys.exit(1)
    video_path = os.path.join("video-footage", video_file)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        sys.exit(1)
    
    fps = cap.get(cv2.CAP_PROP_FPS)

    landmarker = setup_face_landmarker(model_path)

    # process video file: capture frames and run face detection for ROI extraction.
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames.append(frame.copy())

        # for face detection, convert BGR to RGB.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        last_result = landmarker.detect_for_video(mp_image, timestamp)

        if last_result and last_result.face_landmarks:
            process_frame(frame, last_result.face_landmarks[0])

    cap.release()
    
    if not video_frames:
        print("No video frames captured.")
        sys.exit(1)
    
    # convert raw video frames (BGR) to RGB for EVM processing.
    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in video_frames]
    
    # apply EVM (color magnification) to the entire sequence.
    magnified_rgb_frames = mag_colors(rgb_frames, fps, freq_lo=FREQ_LOW,
                                      freq_hi=FREQ_HIGH, level=LEVEL, alpha=ALPHA)
    # convert magnified frames (which are in RGB) back to BGR for display.
    magnified_bgr_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in magnified_rgb_frames]
    
    # extract ROI signal
    if final_roi_coords is None:
        print("ROI was not determined.")
        sys.exit(1)
    raw_signal = extract_roi_signal(video_frames, final_roi_coords)
    evm_signal = extract_roi_signal(magnified_bgr_frames, final_roi_coords)
    
    # display original and EVM processed frames side by side 
    for orig, evm in zip(video_frames, magnified_bgr_frames):
        # combine the original and processed frames horizontally.
        side_by_side = np.hstack((orig, evm))
        cv2.imshow("Original (Left) | EVM Processed (Right)", side_by_side)
        if cv2.waitKey(30) & 0xFF == ord('q'): # q = quit
            break
    cv2.destroyAllWindows()
    
    # plot the ROI signals 
    plt.figure(figsize=(10, 5))
    plt.plot(raw_signal, label="Raw ROI Signal (Green Channel)")
    plt.plot(evm_signal, label="EVM ROI Signal (Green Channel)")
    plt.xlabel("Frame Index")
    plt.ylabel("Average Green Value")
    plt.title("Comparison: Raw vs. Color-Magnified ROI Signals")
    plt.legend()
    plt.tight_layout()
    plt.show()



    

if __name__ == "__main__":
    main()
