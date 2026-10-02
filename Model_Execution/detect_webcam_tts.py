import os
import sys
import cv2
import pyttsx3
import threading # Crucial for non-blocking speech
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = os.path.join('.', 'runs', 'detect', 'train5', 'weights', 'best.pt')
CAMERA_INDEX = 1            # Use 0 for default webcam (try 1 or 2 if 0 fails)
CONFIDENCE_THRESHOLD = 0.25  # Lowered to 0.25 to catch more detections

# Define colors for the screen partitions
LINE_COLOR = (255, 255, 255) # White
TEXT_COLOR = (0, 0, 255)     # Red for warnings/messages 
SUCCESS_COLOR = (0, 255, 0)  # Green for detected object text

# --- Helper Function for Threaded (Non-Blocking) TTS ---
def speak_message_threaded(text):
    """Speaks the text in a separate thread to prevent blocking the main video loop."""
    print(f"TTS: {text}")
    
    # Define the target function for the thread
    def speech_thread():
        # The engine must be initialized and used within its own thread
        try:
            # We initialize a new engine instance for the thread's scope
            engine = pyttsx3.init() 
            engine.say(text)
            # This method runs the speech
            engine.runAndWait() 
        except Exception as e:
            # Catch exceptions like busy drivers or uninitialized speakers
            print(f"TTS Error: Could not speak message. {e}")
        
    # Start the speech in a background thread
    t = threading.Thread(target=speech_thread)
    t.start()

# --- Main Detection Function ---
def detect_on_webcam_pro():
    try:
        model = YOLO(MODEL_PATH)
        print(f"Loaded model from: {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit()

    # Initialize webcam capture
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"Error: Could not open camera with index {CAMERA_INDEX}.")
        sys.exit()
    
    # --- TTS A: Welcome Message ---
    speak_message_threaded("Welcome to the Smart Sight: The real-time object detection system. Starting scan now.")

    # Get initial frame properties
    # Note: These values might change slightly if the camera needs time to initialize fully,
    # but we will rely on them for the partition lines.
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define screen partitions (25%, 50%, 25%)
    W_25 = int(W * 0.25)
    W_75 = int(W * 0.75)
    
    print("\nStarting real-time detection. Press 'q' to exit the window.")
    
    # Track the last spoken message to avoid rapid repetition
    # This prevents the system from shouting "CAR FRONT! CAR FRONT! CAR FRONT!" on every frame
    last_spoken_message = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        frame = cv2.flip(frame, 1) # Flip for selfie view

        # Run detection
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        
        # --- 1. Draw Window Partitions ---
        cv2.line(frame, (W_25, 0), (W_25, H), LINE_COLOR, 2)
        cv2.line(frame, (W_75, 0), (W_75, H), LINE_COLOR, 2)
        
        # List to track unique objects detected in the current frame
        current_frame_messages = []
        
        # --- Process and Annotate Detections ---
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            
            if score > CONFIDENCE_THRESHOLD:
                # --- Get Object Center and Location ---
                x_center = (x1 + x2) / 2
                class_name = results.names[int(class_id)].upper()
                
                # Determine object location based on center x-coordinate
                if x_center < W_25:
                    location = "left side"
                    text_origin = (20, 50) 
                elif x_center > W_75:
                    location = "right side"
                    text_origin = (W_75 + 10, 50) 
                else:
                    location = "front"
                    text_origin = (W_25 + 10, 50) 

                # --- Draw Detection (Bounding Box) ---
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), SUCCESS_COLOR, 3)
                label = f"{class_name} ({score:.2f})"
                cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, SUCCESS_COLOR, 2, cv2.LINE_AA)
                
                # --- 2. On-Screen Message Display ---
                message = f"{class_name} detected in {location}!"
                
                # Create a unique message identifier for the TTS non-repetition logic
                unique_message_id = f"{class_name}_{location}"
                current_frame_messages.append(unique_message_id)
                
                # Display the detected object message on screen
                cv2.putText(frame, message, text_origin,
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT_COLOR, 3, cv2.LINE_AA)
                
                # --- 2. TTS B: Speak Message (Non-Repeating Logic) ---
                # Speak only if this specific detection is NEW or if nothing has been spoken recently
                if unique_message_id != last_spoken_message:
                    speak_message_threaded(f"{class_name} in {location}")
                    last_spoken_message = unique_message_id
        
        # If no objects are detected in the current frame, or the object has moved/changed,
        # reset the last_spoken_message to allow the system to announce the next object.
        if not current_frame_messages:
             last_spoken_message = ""

        # Display the output frame
        cv2.imshow('YOLOv8 Real-Time Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    # Speak goodbye message
    speak_message_threaded("Detection session ended. Goodbye.")
    print("Webcam detection stopped.")

if __name__ == "__main__":
    detect_on_webcam_pro()