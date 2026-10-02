import onnx
import tf2onnx
import tensorflow as tf
import os

# Path to your ONNX model
ONNX_MODEL_PATH = os.path.join('.', 'runs', 'detect', 'train5', 'weights', 'best.onnx')
# Path for the output TFLite model
TFLITE_MODEL_PATH = os.path.join('.', 'runs', 'detect', 'train5', 'weights', 'best_float16.tflite') # or best_int8.tflite

# 1. Load ONNX model
onnx_model = onnx.load(ONNX_MODEL_PATH)

# 2. Convert ONNX to TensorFlow Graph
# This step generates a TensorFlow graph from the ONNX model.
# Specify input names and output names if known (e.g., from Netron inspection)
# For YOLOv8, inputs often 'images', outputs often 'output0'
tf_graph, _ = tf2onnx.convert.from_onnx(onnx_model)

# 3. Create a TensorFlow Lite converter
converter = tf.lite.TFLiteConverter.from_session(tf_graph)

# Optional: Optimize for TFLite (quantization for smaller size/faster inference)
# You can choose 'DEFAULT' for float16 or 'INT8' for integer quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT] # For Float16 quantization
# converter.optimizations = [tf.lite.Optimize.DEFAULT] # For Integer quantization
# converter_input_arrays = ["input_name"] # If you know the input name from ONNX
# converter_output_arrays = ["output_name"] # If you know the output name from ONNX

# Enable experimental new converter
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # Enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # Enable select TensorFlow ops.
]

# Convert to TFLite
tflite_model = converter.convert()

# Save the TFLite model
with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"Model successfully converted to TFLite: {TFLITE_MODEL_PATH}")