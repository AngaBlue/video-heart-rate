import cv2
import os
import numpy as np

def calculate_bpp(video_path):
    # Open the video using OpenCV
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    bitrate = cap.get(cv2.CAP_PROP_BITRATE) * 1_000
    print(f"Height: {height}px")
    print(f"Width: {width}px")
    print(f"Framerate: {fps}/s")
    print(f"Bitrate: {bitrate / 1_000}kb/s")
    
    # Calculate the total number of pixels per second
    pixels_per_second = width * height * fps
    
    # Calculate the BPP
    bpp = bitrate / pixels_per_second
    
    # Release the video capture object
    cap.release()

    return bpp

def calculate_entropy(image):
    """
    Calculate the entropy of an image.
    :param image: Grayscale image (2D numpy array).
    :return: Entropy value.
    """
    # Flatten the image and compute the histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist / hist.sum()  # Normalize the histogram

    # Compute the entropy
    entropy = -np.sum(hist * np.log2(hist + 1e-6))  # Add small epsilon to avoid log(0)
    return entropy

def calculate_video_entropy(video_path):
    """
    Calculate the entropy of a video by evaluating each frame.
    :param video_path: Path to the video file.
    :return: Average entropy of the video.
    """
    # Open the video using OpenCV
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    entropy_values = []

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break  # End of video
        
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate entropy for the current frame
        entropy = calculate_entropy(gray_frame)
        entropy_values.append(entropy)

    # Calculate average entropy for the entire video
    avg_entropy = np.mean(entropy_values)
    print(f"Average Entropy of the Video: {avg_entropy:.4f}")

    # Release the video capture object
    cap.release()

def calculate_noise(frame):
    """
    Calculate the variance (a measure of noise) in a given video frame.
    :param frame: Grayscale video frame (2D numpy array).
    :return: Variance of pixel intensities.
    """
    # Calculate the variance of pixel values in the frame
    variance = np.var(frame)
    return variance

def calculate_video_noise(video_path):
    """
    Calculate the noise (variance) of a video by evaluating each frame.
    :param video_path: Path to the video file.
    :return: Average noise (variance) across the entire video.
    """
    # Open the video using OpenCV
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    noise_values = []

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break  # End of video
        
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate noise (variance) for the current frame
        noise = calculate_noise(gray_frame)
        noise_values.append(noise)

    # Calculate average noise for the entire video
    avg_noise = np.mean(noise_values)
    print(f"Average Noise (Variance) of the Video: {avg_noise:.4f}")

    # Release the video capture object
    cap.release()

def calculate_ns_ratio(frame):
    """
    Calculate the Noise-to-Signal Ratio (NSR) for a given video frame.
    :param frame: Grayscale video frame (2D numpy array).
    :return: NSR value (standard deviation / mean).
    """
    mean_signal = np.mean(frame)
    std_noise = np.std(frame)
    
    # Avoid division by zero
    if mean_signal == 0:
        return 0
    nsr = std_noise / mean_signal
    return nsr

def calculate_video_nsr(video_path):
    """
    Calculate the NSR (Noise-to-Signal Ratio) of a video by evaluating each frame.
    :param video_path: Path to the video file.
    :return: Average NSR across the entire video.
    """
    # Open the video using OpenCV
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    nsr_values = []

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break  # End of video
        
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate NSR for the current frame
        nsr = calculate_ns_ratio(gray_frame)
        nsr_values.append(nsr)

    # Calculate average NSR for the entire video
    avg_nsr = np.mean(nsr_values)
    print(f"Average NSR of the Video: {avg_nsr:.4f}")

    # Release the video capture object
    cap.release()

if __name__ == '__main__':
    video_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'video-footage')

    # Select video file
    print("Select input video file:")
    files = list(os.listdir(video_dir))
    for i, path in enumerate(files):
        print(f"[{i + 1}] {path}")
    choice = int(input().strip()) - 1

    if choice < 0 or choice >= len(files):
        print("Invalid choice, exiting...")
        exit(1)
    
    path = os.path.join(video_dir, files[choice])
    bpp = calculate_bpp(path)
    print(f"BPP: {bpp:.4f}b/p")
    calculate_video_nsr(path)
