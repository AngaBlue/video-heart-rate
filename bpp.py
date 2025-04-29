import cv2
import os

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
