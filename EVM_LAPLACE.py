import os
import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from collections import deque
import sys


# TODO: fix finding path for video and model (doesnt work unless we inside video-heart rate)


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


# signal storage (for ROI extraction demo)
green_signal_forehead = deque(maxlen=500)  # we don't need old frames
green_signal_cheek = deque(maxlen=500)
last_result = None

# EVM parameters
AMPLIFY_Y = 1  # luminance   
AMPLIFY_I = 40 # R and G 
AMPLIFY_Q = 40 # B and R 
LAMBDA_CUTOFF = 1000 # spatial cutoff
LEVEL = 4             # number of downscaling steps in the Gaussian pyramid (use 3-5)
FREQ_LOW = 0.8  # 42 BPM
FREQ_HIGH = 1 # 120 BPM (MIT USES 0.5 to 3 HZ, best results 0.8 - 1)

# why is this doingggg 
ALPHA = 1.0    

###########################
#   Color Magnification   # CITATION: https://hbenbel.github.io/blog/evm/ AND https://people.csail.mit.edu/mrub/evm/
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



def laplacian_video(video_stack, level):
    """
    Create Laplacian pyramid for a video stack per level: [num_frames][H, W, C].
    """

    num_frames = len(video_stack)

    # compute the shapes of each level using the first frame
    g_pyramid = [video_stack[0].copy()]
    for _ in range(1, level):
        g_pyramid.append(cv2.pyrDown(g_pyramid[-1]))

    level_shapes = [(img.shape[0], img.shape[1], img.shape[2]) for img in g_pyramid]

    # allocate NumPy arrays for each level
    laplace_video = [np.empty((num_frames, h, w, c), dtype=np.float32) for (h, w, c) in level_shapes]

    for i, frame in enumerate(video_stack):
        gaussian = frame.copy()
        g_pyramid = [gaussian]
        for _ in range(1, level):
            gaussian = cv2.pyrDown(gaussian)
            g_pyramid.append(gaussian)

        for n in range(level - 1):
            size = (g_pyramid[n].shape[1], g_pyramid[n].shape[0])
            upsampled = cv2.pyrUp(g_pyramid[n + 1], dstsize=size)
            laplacian = g_pyramid[n] - upsampled
            laplace_video[n][i] = laplacian

        laplace_video[-1][i] = g_pyramid[-1]  # add coarsest level for reconstruction

    return laplace_video




    
def bandpass_butterworth(signal, fps, freq_lo, freq_high, order=1):
    '''
    Calculate the Butterworth bandpass filter, input = [T, N]
    '''
    nyquist = 0.5 * fps
    low = freq_lo / nyquist
    high = freq_high / nyquist
    b, a = sp.butter(order, [low, high], btype='band')
    filtered = sp.filtfilt(b, a, signal, axis=0) # eliminate delay/lag by filtering fwd and bwd
    return filtered


"""
from MIT website

To process an input video by Eulerian video magnification, there
are four steps a user needs to take: 
(1) select a temporal bandpass filter; 
(2) select an amplification factor, α; 
(3) select a spatial frequency cutoff (specified by spatial wavelength, λc) beyond which
an attenuated version of α is used; and 
(4) select the form of the attenuation for α—either force α to zero for all λ < λc, or linearly
scale α down to zero. The frequency band of interest can be chosen automatically in some cases, but it is often important for users
to be able to control the frequency band corresponding to their application. 

"""


def apply_butterworth(laplace_video_pyramid, fps, vidWidth, vidHeight, freq_lo, freq_hi, level, lambda_cutoff, alpha):
    """
    Apply temporal Butterworth bandpass filter across time for each pyramid level.
    Amplify motion based on spatial wavelength and level-specific alpha.

    Parameters:
        laplace_video_pyramid: list of NumPy arrays, one per pyramid level, shape [T, H, W, C]
        fps: frame rate
        vidWidth, vidHeight: resolution of the original video
        freq_lo, freq_hi: temporal bandpass range
        level: number of pyramid levels
        lambda_cutoff: spatial wavelength threshold
        alpha: amplification factor
    Returns:
        filtered_video: list of same shape as input, but temporally filtered and amplified
    """

    filtered_video = []

    lambda_level = (vidWidth ** 2 + vidHeight ** 2) ** 0.5
    delta = (lambda_cutoff / 8) / (1 + alpha)
    """ SHAPE
    301 592 528 3
    301 296 264 3
    301 148 132 3
    301 74 66 3
    
    """
    for n in range(level):
        current_level = laplace_video_pyramid[n]  # shape: [T, H, W, C]
        T, H, W, C = current_level.shape

        print(T, H, W, C)
    return

    
    
    '''for n in range(level):
        current_level = laplace_video_pyramid[n]  # shape: [T, H, W, C]
        T, H, W, C = current_level.shape

        current_alpha = (lambda_level / (8 * delta)) - 1
        current_alpha = min(alpha, current_alpha)

        # bandpass filter each channel independently
        filtered_level = np.zeros_like(current_level)

        for c in range(C):
            signal = current_level[:, :, :, c]  # shape [T, H, W]
            signal_reshaped = signal.reshape(T, -1)  # reshape to [T, H*W] for filtering

            filtered = bandpass_butterworth(signal_reshaped, fps, freq_lo, freq_hi)
            if 0 < n < level - 1:
                filtered *= current_alpha
            else:
                filtered = 0
        
            # reshape back
            filtered_level[:, :, :, c] = filtered.reshape(T, H, W)

            # attenuation for chrominance (SCALE_I, SCALE_Y)
            # if c > 0:
            #     filtered_level[:, :, :, c] *= attenuation

        filtered_video.append(filtered_level)
        lambda_level /= 2  # spatial wavelength halves each level

    return filtered_video'''


    











def reconstruct_video(filtered_video, level):
    """
    Reconstructs a video from its filtered Laplacian pyramid of shape [num_frames, H, W, C]
    """

    pass





def mag_colors(rgb_frames, fps, vidWidth, vidHeight):
    """
    perform EVM for color-based amplification
    """
    # initialise variables
    num_frames = len(rgb_frames)

    # convert frames to YIQ colorspace (normalize by 255)
    yiq_frames = [rgb2yiq(frame / 255.0) for frame in rgb_frames]

    # build Laplacian pyramid for each frame.
    laplace_video_pyramid =  laplacian_video(yiq_frames, level=LEVEL)

    print("laplacian pyramid complete")

    filtered_video = apply_butterworth(laplace_video_pyramid, fps, vidWidth, vidHeight,
                                       freq_lo=FREQ_LOW, freq_hi=FREQ_HIGH, level=LEVEL, lambda_cutoff=LAMBDA_CUTOFF, alpha=ALPHA)
    
    print("butterworth filter complete")

    #result = reconstruct_video(filtered_video, level=LEVEL)

    #print("reconstruction complete")

    """
    I FEEL LIKE U SHUD DO THIS ALL IN ONE!!!!
    u do too much one at a time, reconstruct and amplify and add all in one function
    
    """
 
    



    # cutoff wrong values
    #rgb_video[rgb_video < 0] = 0
    #rgb_video[rgb_video > 255] = 255


    return 
    
    


###############################
#   Face Detection and ROI    #
###############################



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
    For a given frame and detected face landmarks, compute the face bounding box and define ROI
    """
    global final_roi_coords, raw_roi_signal
    h, w, _ = frame_bgr.shape
    x_vals = [lm.x for lm in landmarks]
    y_vals = [lm.y for lm in landmarks]
    bb_x1 = int(min(x_vals) * w)
    bb_y1 = int(min(y_vals) * h)
    bb_x2 = int(max(x_vals) * w)
    bb_y2 = int(max(y_vals) * h)
    
    # cheek: 0.15, 0.4, 0.65
    # forehead: 0.25, 0.00, 0.25
    roi_coords = get_roi_coords(bb_x1, bb_y1, bb_x2, bb_y2,
                                horizontal_ratio=0.15,
                                top_ratio=0.4,
                                bottom_ratio=0.65)
    
    final_roi_coords = roi_coords
    roi = frame_bgr[roi_coords[1]:roi_coords[3], roi_coords[0]:roi_coords[2]]
    raw_roi_signal.append(roi)


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




###############################
#          Calculations       #
###############################


# TODO:does find peaks work???
def calculate_bpm(signal, fps):
    """
    Basic BPM estiamation, not sure if this is actually right for our evm heart rate system
    """

    # convert to NumPy array and remove 0 Hz component 
    sig = np.asarray(signal, dtype=float) 
    sig -= np.mean(sig) # so peak detection isn’t biased by a wandering baseline

    # only count strong prominent peaks
    prominence = np.std(sig) * 0.5 # TODO: tweak this

    # detect local maxima at least 0.4 s apart  
    min_distance = int(0.4 * fps)
    peaks, _ = sp.find_peaks(sig, distance=min_distance, prominence=prominence)

    # time (in seconds) between successive peaks
    intervals = np.diff(peaks) / fps

    return 60.0 / intervals.mean()





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
    
    # process video
    fps = cap.get(cv2.CAP_PROP_FPS)
    vidWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vidHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
    magnified_rgb_frames = mag_colors(rgb_frames, fps, vidWidth, vidHeight)
    
    # convert magnified frames (which are in RGB) back to BGR for display.
    '''magnified_bgr_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in magnified_rgb_frames]
    
    # extract ROI signal
    if final_roi_coords is None:
        print("ROI was not determined.")
        sys.exit(1)
    raw_signal = extract_roi_signal(video_frames, final_roi_coords)
    evm_signal = extract_roi_signal(magnified_bgr_frames, final_roi_coords)

    # calculate heart rate bpm 
    print(f"raw signal: {calculate_bpm(raw_signal, fps):.1f} bpm")
    print(f"magnified signal: {calculate_bpm(evm_signal, fps):.1f} bpm")


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
    plt.show()'''


    
main()

