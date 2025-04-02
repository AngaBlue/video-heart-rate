import mediapipe as mp
import cv2 as cv
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from collections import deque
from scipy.signal import butter, filtfilt, detrend, find_peaks


# load mediapipe face detection/landmark model
model_path = "/Users/ahila/Desktop/fyp - video heart rate monitor/face_landmarker.task"

# MediaPipe vision modules (thank you daddy google)
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# store latest detection result
last_result = None


# graphing the green channel signal
green_signal_forehead = deque(maxlen=300) # we dont need old frames
green_signal_cheek = deque(maxlen=300)


plt.ion()  #  interactive mode
fig, ax = plt.subplots()
line1, = ax.plot([], [], label="Forehead")
line2, = ax.plot([], [], label="Cheek")
ax.set_ylim(-2, 2) # since we normalized and filtered signal, keep it centered around 0 
ax.set_xlim(0, 300) # 10-second window at 30 fps
ax.set_title("Green Channel Signal")
ax.set_xlabel("Frame")
ax.set_ylabel("Filtered Green Value")
ax.legend()


# bpm is 60/period
def bandpass_filter(signal, fs=30, low=0.7, high=4.0, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)



# callback function to handle the asynchronous results
def get_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):  # type: ignore
    global last_result
    last_result = result

    # print('face detector result: {}'.format(result))


def get_roi(horizontal_ratio, top_ratio, bottom_ratio):

    roi_y1 = int(bb_y1 + top_ratio * (bb_y2 - bb_y1))
    roi_y2 = int(bb_y1 + bottom_ratio * (bb_y2 - bb_y1))

    roi_x1 = int(bb_x1 + horizontal_ratio * (bb_x2 - bb_x1))
    roi_x2 = int(bb_x2 - horizontal_ratio * (bb_x2 - bb_x1))

    # draw roi
    cv.rectangle(frame_bgr, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

    return roi_x1, roi_y1, roi_x2, roi_y2


# webcam
cam = cv.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open camera")
else:
    # MediaPipe face landmarker task
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=get_result,  # asynchronous callback, update last_result whenever a new detection is available
        num_faces=1
    )
    
    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame_bgr = cam.read()
            if not ret:
                print("Error: Failed to capture image")
                break

            # convert to RGB for MediaPipe
            frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            frame_timestamp = int(time.time() * 1000)

            # asynchronous landmark detection
            landmarker.detect_async(mp_image, frame_timestamp)

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

                # draw full face bounding box
                cv.rectangle(frame_bgr, (bb_x1, bb_y1), (bb_x2, bb_y2), (0, 255, 0), 2)

                # forehead ROI
                f_x1, f_y1, f_x2, f_y2 = get_roi(horizontal_ratio=0.25, top_ratio=0.00, bottom_ratio=0.25)

                # mid-face (cheek) ROI
                c_x1, c_y1, c_x2, c_y2 = get_roi(horizontal_ratio=0.15, top_ratio=0.4, bottom_ratio=0.65)

                # extract average from green channel (Beer-Lambert Law)
                roi_forehead = frame_bgr[f_y1:f_y2, f_x1:f_x2]
                avg_green_forehead = np.mean(roi_forehead[:, :, 1])

                roi_cheek = frame_bgr[c_y1:c_y2, c_x1:c_x2]
                avg_green_cheek = np.mean(roi_cheek[:, :, 1])

                green_signal_forehead.append(avg_green_forehead)
                green_signal_cheek.append(avg_green_cheek)

                # display the filtered green intensity signal
                if len(green_signal_forehead) >= 30: # ensure signal is long enough to filter

                    # remove slow-changing noise e.g lighting, exposure
                    g_forehead = detrend(np.array(green_signal_forehead))
                    g_cheek = detrend(np.array(green_signal_cheek))

                    # normalise using z score (mean=0, sd=1) to remove outliers
                    g_forehead = (g_forehead - np.mean(g_forehead)) / np.std(g_forehead)
                    g_cheek = (g_cheek - np.mean(g_cheek)) / np.std(g_cheek)

                    # keep frequencies within heart rate range (0.7 - 4Hz)
                    g_forehead_filtered = bandpass_filter(g_forehead)
                    g_cheek_filtered = bandpass_filter(g_cheek)

            
                    # update plot
                    line1.set_ydata(g_forehead_filtered)
                    line1.set_xdata(range(len(g_forehead_filtered)))
                    line2.set_ydata(g_cheek_filtered)
                    line2.set_xdata(range(len(g_cheek_filtered)))

                    # estimate BPM from filtered forehead signal
                    peaks, _ = find_peaks(g_forehead_filtered, distance=15)
                    bpm = len(peaks) * (60 / (len(g_forehead_filtered) / 30))
                    print(f"Estimated BPM: {bpm:.1f}")

                    # update plot limits
                    ax.relim()
                    ax.autoscale_view()
                    plt.pause(0.001)

            # display your beautiful face
            cv.imshow('Face Landmarker with ROIs', frame_bgr)

            # Break loop on 'q'
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cam.release()
    cv.destroyAllWindows()





