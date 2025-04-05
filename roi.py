import os
# Suppress TensorFlow logs for macos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress absl logs used by MediaPipe for macos
os.environ['GLOG_minloglevel'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'


import mediapipe as mp
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from collections import deque
import time


# signal storage
green_signal_forehead = deque(maxlen=300) # we dont need old frames
green_signal_cheek = deque(maxlen=300)
last_result = None


# MediaPipe setup (thank u papa google)
BaseOptions = mp.tasks.BaseOptions # load model
FaceLandmarker = mp.tasks.vision.FaceLandmarker # create landmarker object
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions # configure landmarker
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode # select mode, e.g VIDEO or LIVE-STREAM


# TODO!  fix params
def bandpass_filter(signal, fs=30, low=0.7, high=4.0, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)


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


# select VIDEO or LIVESTREAM mode
def select_input():
    input_choice = input("select input: [1] webcam -- [2] video-footage light -- [3] video-footage dark: ").strip()
    if input_choice == "1":
        cam = cv.VideoCapture(0)
        if not cam.isOpened():
            print("error: could not open webcam.")
            exit()
        return cam, VisionRunningMode.LIVE_STREAM, True
    elif input_choice == "2":
        path = os.path.join("video-footage", "light_skin.mp4")
        return cv.VideoCapture(path), VisionRunningMode.VIDEO, False
    elif input_choice == "3":
        path = os.path.join("video-footage", "dark_skin.mp4")
        return cv.VideoCapture(path), VisionRunningMode.VIDEO, False
    else:
        print("invalid choice.")
        exit()



def get_roi_coords(bb_x1, bb_y1, bb_x2, bb_y2, horizontal_ratio, top_ratio, bottom_ratio, frame_bgr):
    roi_y1 = int(bb_y1 + top_ratio * (bb_y2 - bb_y1))
    roi_y2 = int(bb_y1 + bottom_ratio * (bb_y2 - bb_y1))
    roi_x1 = int(bb_x1 + horizontal_ratio * (bb_x2 - bb_x1))
    roi_x2 = int(bb_x2 - horizontal_ratio * (bb_x2 - bb_x1))
    cv.rectangle(frame_bgr, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
    return roi_x1, roi_y1, roi_x2, roi_y2


def extract_green_channel(roi):
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

    avg_green_forehead = extract_green_channel(roi_forehead)
    avg_green_cheek = extract_green_channel(roi_cheek)

    green_signal_forehead.append(avg_green_forehead)
    green_signal_cheek.append(avg_green_cheek)



def main():

    print("PRESS q to quit -- PRESS spacebar to pause")

    # important variables
    global last_result # last detected frame
    paused = False # boolean flag for pause functionality
    last_frame = None # copy of last detected frame

    # setup plot
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

    # select input and mode
    cam, running_mode, use_callback = select_input()
    landmarker = setup_face_landmarker(model_path, running_mode, get_result if use_callback else None)

    with landmarker:
        while True:
            if not paused:    
                ret, frame_bgr = cam.read()

                # if we hit the end of our video footage (last frame)
                if not ret:
                    print("End of stream reached — freezing displays. Press 'q' to quit.")
                    # disable interactive mode to freeze graph at last result
                    plt.ioff()
                    plt.show()  # display the final plot
                    # loop to freeze the video window at the last frame.
                    while True:
                        cv.imshow("face landmarker with rois", last_frame)
                        key = cv.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                    break  # break out of the main loop once 'q' is pressed

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
                    process_frame(frame_bgr, last_result.face_landmarks[0])

                    # ensure signal is long enough to filter
                    if len(green_signal_forehead) >= 30:


                        # TODO!

                        '''
                        EVM PROCESS

                        - get video frames in YIQ colorspace
                        - obtain single gaussian pyramid level of each frame
                        - temporal bandpass filter to obtain heart rate between 0.83 to 1.0 Hz
                        - magnify filtered pyramid levels back
                        - add magnified pyramid levels back to original frames
                        - convert back to RGB / BGR color space to display 

                        '''



                        update_plot(green_signal_forehead, green_signal_cheek, line1, line2, ax)

            # if we hit pause, use last captured frame
            else:
                frame_bgr = last_frame
            
            # display frame with roi 
            cv.imshow("face landmarker with rois", frame_bgr)
            
            key = cv.waitKey(1) & 0xFF # wait 1 ms, then extract keycode
            # q = quit
            if key == ord('q'):
                break
            # TODO! when paused, ROI dissapear
            # spacebar = pause  must be clicked on open-cv frame window (face) display 
            if key == ord(' '):
                paused = not paused


    cam.release()
    cv.destroyAllWindows()


main()
'''if __name__ == "__main__":
    main()'''











