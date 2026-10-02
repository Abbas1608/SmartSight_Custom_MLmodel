import os
from ultralytics import YOLO

# Path to your trained model
MODEL_PATH = os.path.join('.', 'runs', 'detect', 'train5', 'weights', 'best.pt')

# Load your model
model = YOLO(MODEL_PATH)

# Export the model to ONNX format
# The 'imgsz' (image size) must match the size you trained with (e.g., 640 for YOLOv8)
# The 'half' argument can reduce model size/inference time if your device supports FP16
model.export(format='onnx', imgsz=640, half=True) 

print(f"Model exported to ONNX. Check the 'runs/detect/train5/weights' folder for 'best.onnx'.")