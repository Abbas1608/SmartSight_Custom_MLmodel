import os
import cv2
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = os.path.join('.', 'runs', 'detect', 'train5', 'weights', 'best.pt')
VIDEO_FILE = 'Video_Generation_With_More_Objects.mp4' # Ensure this video file exists in the directory
CONFIDENCE_THRESHOLD = 0.25  # Lowered to 0.25 to catch more detections
OUTPUT_VIDEO_FILE = os.path.splitext(VIDEO_FILE)[0] + '_detected.avi'

# --- Main Function ---
def detect_on_video():
    try:
        model = YOLO(MODEL_PATH)
        print(f"Loaded model from: {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {VIDEO_FILE}")
        return

    # Get video properties for VideoWriter
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize VideoWriter (using MJPG for wider compatibility)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, FPS, (W, H))
    
    print(f"\nStarting video detection and saving to: {OUTPUT_VIDEO_FILE}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection. Conf is set to the threshold.
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

        # Use the model's built-in plotting utility
        annotated_frame = results.plot()
        
        out.write(annotated_frame)

    # Cleanup
    cap.release()
    out.release()
    print("Video processing finished.")

if __name__ == "__main__":
    detect_on_video()