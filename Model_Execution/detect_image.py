import os
import cv2
from ultralytics import YOLO

# --- Configuration ---
# Path to your successfully trained model (from the train5 run)
MODEL_PATH = os.path.join('.', 'runs', 'detect', 'train5', 'weights', 'best.pt')
# Name of the image file to test (Ensure this file is in the same directory!)
IMAGE_FILE = 'Abbas_image.png' 
CONFIDENCE_THRESHOLD = 0.25  # Lowered to 0.25 to catch more detections

# --- Main Function ---
def detect_on_image():
    # Load the YOLO model
    try:
        model = YOLO(MODEL_PATH)
        print(f"Loaded model from: {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check the model path and file integrity.")
        return

    # Load the image
    image_path = os.path.join(IMAGE_FILE)
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Error: Could not load image file at {image_path}. Check the file name and path.")
        return

    # Run detection with the specified confidence threshold
    # verbose=False suppresses frame-by-frame console output
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

    # Use the model's built-in plotting utility to draw boxes and labels
    annotated_frame = results.plot()

    # Define output path
    output_dir = os.path.join('runs', 'test_results', 'images')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'detected_' + IMAGE_FILE)

    # Save the annotated image
    cv2.imwrite(output_path, annotated_frame)
    print(f"\nDetection complete. Annotated image saved to: {output_path}")

if __name__ == "__main__":
    detect_on_image()