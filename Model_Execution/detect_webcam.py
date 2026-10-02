import sys
import cv2
import os
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = os.path.join('.', 'runs', 'detect', 'train5', 'weights', 'best.pt')
CAMERA_INDEX = 0         
CONFIDENCE_THRESHOLD = 0.15  # Lowered to 0.25 to catch more detections

# --- Main Function ---
def detect_on_webcam():
    try:
        model = YOLO(MODEL_PATH)
        print(f"Loaded model from: {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit()

    # Initialize webcam capture
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW) # CAP_DSHOW for Windows stability

    if not cap.isOpened():
        print(f"Error: Could not open camera with index {CAMERA_INDEX}.")
        print("Try changing the CAMERA_INDEX to 1 or 2.")
        sys.exit()

    print("\nStarting real-time detection. Press 'q' to exit the window.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        # Flip for a "selfie" view (optional)
        frame = cv2.flip(frame, 1)

        # Run detection. Conf is set to the threshold.
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

        # Use the model's built-in plotting utility to draw boxes and labels
        # This handles drawing ALL detected objects automatically
        annotated_frame = results.plot()

        # Display the output frame
        cv2.imshow('YOLOv8 Real-Time Detection (train5)', annotated_frame)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam detection stopped.")

if __name__ == "__main__":
    detect_on_webcam()