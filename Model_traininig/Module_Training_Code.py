from ultralytics import YOLO

# 1. Load the COCO pre-trained weights to get initial knowledge (Transfer Learning)
# This is the crucial step to fix your low mAP score.
model = YOLO("yolov8n.pt") 

# 2. Re-train the model for 100 epochs using your custom data
# The model will fine-tune the COCO knowledge to your 12 custom classes.
# We are increasing the epochs from 12 to 100 to allow sufficient learning time.
results = model.train(data="config.yaml", epochs=100) 

# Results saved to C:\Users\MOHD ABBAS\OneDrive\Desktop\Yolov8n_custom_Model\runs\detect\train5