import mediapipe as mp
import cv2 as cv
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import butter, filtfilt, detrend, find_peaks
from requests import get
import os


# download model once
model_url = "https://raw.githubusercontent.com/AngaBlue/video-heart-rate/main/face_landmarker.task"
model_path = "face_landmarker.task"

if not os.path.exists(model_path):
    response = get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# signal storage with deque (we dont need old frames)
green_signal_forehead = deque(maxlen=300)
green_signal_cheek = deque(maxlen=300)

# plot setup
plt.ion()
fig, ax = plt.subplots()
line1, = ax.plot([], [], label="Forehead")
line2, = ax.plot([], [], label="Cheek")
ax.set_title("Green Channel Signal")
ax.set_xlabel("Frame")
ax.set_ylabel("Filtered Green Value")
ax.legend()


# bandpass filter TODO: what values do we need?
def bandpass_filter(signal, fs=30, low=0.7, high=4.0, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)


last_result = None # stores most recent frame detected
# async callback
def get_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int): #type: ignore
    global last_result
    last_result = result



def get_roi(horizontal_ratio, top_ratio, bottom_ratio):
    roi_y1 = int(bb_y1 + top_ratio * (bb_y2 - bb_y1))
    roi_y2 = int(bb_y1 + bottom_ratio * (bb_y2 - bb_y1))
    roi_x1 = int(bb_x1 + horizontal_ratio * (bb_x2 - bb_x1))
    roi_x2 = int(bb_x2 - horizontal_ratio * (bb_x2 - bb_x1))
    cv.rectangle(frame_bgr, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
    return roi_x1, roi_y1, roi_x2, roi_y2



# input selection, adjust MediaPipe parameters accordingly
input_choice = input("Select input mode: [1] Webcam, [2] Video file: Light,  [3] Video file: Dark").strip()
# live stream (web-cam) option
if input_choice == "1":
    running_mode = VisionRunningMode.LIVE_STREAM
    use_callback = True
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open input.")
        exit()
# pre-saved video for light skin test subject
elif input_choice == "2":
    running_mode = VisionRunningMode.VIDEO
    use_callback = False
    video_path = "todo"
    cam = cv.VideoCapture(video_path)
# pre-saved video for dark skin test subject
elif input_choice == "3":
    running_mode = VisionRunningMode.VIDEO
    use_callback = False
    video_path = "todo"
    cam = cv.VideoCapture(video_path)
else:
    print("Invalid choice.")
    exit()



# LIVE STREAM option requires
if use_callback:
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_path),
        running_mode=running_mode,
        result_callback=get_result,
        num_faces=1
    )
else: # we dont need async callback for VIDEO option
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_path),
        running_mode=running_mode,
        num_faces=1
    )

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        # captures the latest frame for video or webcam
        ret, frame_bgr = cam.read() 
        if not ret:
            print("End of stream or failed to capture frame.")
            break
        
        # convert to RGB for MediaPipe
        frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # livestream
        if use_callback:
            frame_timestamp = int(time.time() * 1000)
            # asynchronous landmark detection
            landmarker.detect_async(mp_image, frame_timestamp)
        # pre-saved video footage
        else:
            frame_timestamp = int(cam.get(cv.CAP_PROP_POS_MSEC))
            result = landmarker.detect(mp_image)
            last_result = result  # mimic callback behavior

        # reading from the global callback: last_result
        if last_result and last_result.face_landmarks:

            face_landmarks = last_result.face_landmarks[0]

            # get full face bounding box
            x_vals = [lm.x for lm in face_landmarks]
            y_vals = [lm.y for lm in face_landmarks]
            min_x, max_x, min_y, max_y = min(x_vals), max(x_vals), min(y_vals), max(y_vals)
            h, w, _ = frame_bgr.shape

            #  normalized coords to pixel coords
            bb_x1, bb_y1 = int(min_x * w), int(min_y * h)
            bb_x2, bb_y2 = int(max_x * w), int(max_y * h)

            cv.rectangle(frame_bgr, (bb_x1, bb_y1), (bb_x2, bb_y2), (0, 255, 0), 2)

            # forehead ROI
            f_x1, f_y1, f_x2, f_y2 = get_roi(horizontal_ratio=0.25, top_ratio=0.00, bottom_ratio=0.25)

            # mid-face (cheek) ROI
            c_x1, c_y1, c_x2, c_y2 = get_roi(horizontal_ratio=0.15, top_ratio=0.4, bottom_ratio=0.65)

            # TODO! this stuff can all be in a function for "plotting signal i think"
            roi_forehead = frame_bgr[f_y1:f_y2, f_x1:f_x2]
            avg_green_forehead = np.mean(roi_forehead[:, :, 1])

            roi_cheek = frame_bgr[c_y1:c_y2, c_x1:c_x2]
            avg_green_cheek = np.mean(roi_cheek[:, :, 1])

            green_signal_forehead.append(avg_green_forehead)
            green_signal_cheek.append(avg_green_cheek)

            # ensure signal is long enough to undergo filtering and processing
            if len(green_signal_forehead) >= 30:
                
                # TODO! how to filter?? perform EVM
                

                g_forehead = green_signal_forehead
                g_cheek = green_signal_cheek

                # update plot
                line1.set_ydata(g_forehead)
                line1.set_xdata(range(len(g_forehead)))
                line2.set_ydata(g_cheek)
                line2.set_xdata(range(len(g_cheek)))

                # recalculate limits and update view
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.001)

        # display frame
        cv.imshow('Face Landmarker with ROIs', frame_bgr)

        # quit = q
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv.destroyAllWindows()











