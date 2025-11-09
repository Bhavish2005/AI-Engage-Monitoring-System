# from keras.preprocessing.image import img_to_array
# import cv2
# import imutils
# from keras.models import load_model
# import numpy as np
#
# # --- 1. Load Models & Configuration ---
#
# # Load Haar cascades for face and eye detection
# face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_eye.xml')
#
# # Path to your trained emotion model
# video_emotion_model_path = 'model_num.hdf5'
#
# # Load the emotion classifier model
# try:
#     emotion_classifier = load_model(video_emotion_model_path, compile=False)
# except IOError:
#     print(f"Error: Model file not found at {video_emotion_model_path}")
#     print("Please make sure 'model_num.hdf5' is in the same folder as this script.")
#     exit()
#
# # List of emotion labels
# EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]
#
# # --- 2. Initialize Video Capture (Webcam Only) ---
#
# print("Starting webcam...")
# # Use 0 for the default webcam
# cap = cv2.VideoCapture(0)
#
# # Check if the webcam opened successfully
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     print("Please check if your webcam is connected and not in use by another application.")
#     exit()
#
# # Create windows to display the output
# cv2.namedWindow('Student Attention Detector')
# cv2.namedWindow('Face Emotion Probabilities using AI')
#
# # Initialize map for emotion tracking (if needed later)
# emotions_map = {
#     "angry": 0, "disgust": 0, "fear": 0, "happy": 0,
#     "sad": 0, "surprised": 0, "neutral": 0
# }
#
# current_second = 0  # This variable is used in the text, so we keep it
#
# # --- 3. Start Real-Time Processing Loop ---
#
# while True:
#     try:
#         # Read a frame from the webcam
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to grab frame from webcam.")
#             break
#
#         # Resize frame for faster processing and flip
#         frame = imutils.resize(frame, width=400)
#         frame = cv2.flip(frame, 1)  # Flip horizontally for a mirror effect
#
#         # Convert to grayscale for face detection
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#         # Detect faces in the grayscale frame
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
#                                               flags=cv2.CASCADE_SCALE_IMAGE)
#
#         # Create a blank canvas to show emotion probabilities
#         canvas = np.zeros((350, 400, 3), dtype="uint8")
#
#         # --- 4. Process Detected Faces ---
#
#         if (len(faces) == 0):
#             # No face detected
#             attentive = False
#             cv2.putText(frame, "Not-Attentive (student unavailable)", (10, 23),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
#
#         for (x, y, w, h) in faces:
#             # Draw a rectangle around the detected face
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#
#             # Get the Region of Interest (ROI) for both gray and color
#             roi_gray = gray[y:y + h, x:x + w]
#             roi_color = frame[y:y + h, x:x + w]
#
#             # --- Eye Detection (for Attentiveness) ---
#             eyes = eye_cascade.detectMultiScale(roi_gray)
#             for (ex, ey, ew, eh) in eyes[:2]:  # Draw on first 2 eyes
#                 cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
#
#             # --- Emotion Prediction ---
#             # Prepare the face ROI for the emotion model
#             roi_emotion = cv2.resize(roi_gray, (48, 48))
#             roi_emotion = roi_emotion.astype("float") / 255.0
#             roi_emotion = img_to_array(roi_emotion)
#             roi_emotion = np.expand_dims(roi_emotion, axis=0)
#
#             # Predict emotions (NO CHANGE to model arguments)
#             preds = emotion_classifier.predict(roi_emotion)[0]
#             label = EMOTIONS[preds.argmax()]
#             emotions_map[label] += 1
#
#             # --- Draw Probability Bars on Canvas ---
#             for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
#                 text = "{}: {:.2f}%".format(emotion, prob * 100)
#                 w = int(prob * 300)
#                 cv2.rectangle(canvas, (7, (i * 35) + 5),
#                               (w, (i * 35) + 35), (0, 0, 255), -1)
#                 cv2.putText(canvas, text, (10, (i * 35) + 23),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
#
#             # --- Determine Attentiveness ---
#             attentive = False
#             if (len(eyes) >= 1):  # At least one eye detected
#                 attentive = True
#
#             # --- Set the label text based on attentiveness (MODIFIED LOGIC) ---
#             if (attentive):
#                 label_text = "Attentive (" + label + ")"
#                 color = (0, 255, 0)  # Green
#             else:
#                 # If not attentive (eyes closed), check for 'neutral' emotion
#                 if label == "neutral":
#                     label_text = "Not-Attentive (Sleeping)"
#                 else:
#                     label_text = "Not-Attentive (" + label + ")"
#                 color = (0, 0, 255)  # Red
#             # --- End of modified logic ---
#
#             cv2.putText(frame, label_text, (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
#
#         # --- 5. Display the Results ---
#         cv2.imshow('Student Attention Detector', frame)
#         cv2.imshow('Face Emotion Probabilities using AI', canvas)
#
#         # Check for 'q' key to quit the loop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         break  # Exit the loop on an unexpected error
#
# # --- 6. Cleanup ---
# print("Exiting and cleaning up...")
# cap.release()
# cv2.destroyAllWindows()

from keras.preprocessing.image import img_to_array
import cv2
import imutils
from keras.models import load_model
import numpy as np

# --- NEW Imports ---
import time
import threading
from datetime import datetime
from playsound import playsound
import os

# --- 1. Load Models & Configuration ---

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_eye.xml')

# Path to your trained emotion model
video_emotion_model_path = 'model_num.hdf5'

# Load the emotion classifier model
try:
    emotion_classifier = load_model(video_emotion_model_path, compile=False)
except IOError:
    print(f"Error: Model file not found at {video_emotion_model_path}")
    print("Please make sure 'model_num.hdf5' is in the same folder as this script.")
    exit()

# List of emotion labels
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]

# --- NEW: Alert Configuration ---
ALERT_SOUND_FILE = "mixkit-urgent-simple-tone-loop-2976.wav"  # Make sure you have this file!
NON_ATTENTIVE_TIME_LIMIT = 5  # Seconds
SNAPSHOT_DIR = "snapshots"

# --- NEW: State variables for the timer ---
non_attentive_start_time = None
alert_triggered = False

# --- NEW: Create snapshots directory ---
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


# --- NEW: Function to play sound in a separate thread ---
def play_alert():
    try:
        playsound(ALERT_SOUND_FILE)
    except Exception as e:
        print(f"Error playing sound: {e}")
        print("Make sure 'alert.wav' (or .mp3) is in the folder and 'playsound' is installed.")


# --- 2. Initialize Video Capture (Webcam Only) ---

print("Starting webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create windows to display the output
cv2.namedWindow('Student Attention Detector')
cv2.namedWindow('Face Emotion Probabilities using AI')

emotions_map = {
    "angry": 0, "disgust": 0, "fear": 0, "happy": 0,
    "sad": 0, "surprised": 0, "neutral": 0
}

# --- 3. Start Real-Time Processing Loop ---

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from webcam.")
            break

        frame = imutils.resize(frame, width=400)
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                              flags=cv2.CASCADE_SCALE_IMAGE)
        canvas = np.zeros((350, 400, 3), dtype="uint8")

        # --- MODIFIED: Track overall frame attention ---
        is_frame_attentive = False  # Assume not attentive for this frame

        # --- 4. Process Detected Faces ---

        if (len(faces) == 0):
            # No face detected
            is_frame_attentive = False  # --- MODIFIED ---
            cv2.putText(frame, "Not-Attentive (student unavailable)", (10, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # --- Eye Detection (for Attentiveness) ---
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes[:2]:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # --- Emotion Prediction ---
            roi_emotion = cv2.resize(roi_gray, (48, 48))
            roi_emotion = roi_emotion.astype("float") / 255.0
            roi_emotion = img_to_array(roi_emotion)
            roi_emotion = np.expand_dims(roi_emotion, axis=0)

            preds = emotion_classifier.predict(roi_emotion)[0]
            label = EMOTIONS[preds.argmax()]
            emotions_map[label] += 1

            # --- Draw Probability Bars on Canvas ---
            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                text = "{}: {:.2f}%".format(emotion, prob * 100)
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5),
                              (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

            # --- Determine Attentiveness ---
            attentive = False
            if (len(eyes) >= 1):  # At least one eye detected
                attentive = True
                is_frame_attentive = True  # --- MODIFIED ---

            # Set the label text
            if (attentive):
                label_text = "Attentive (" + label + ")"
                color = (0, 255, 0)  # Green
            else:
                if label == "neutral":
                    label_text = "Not-Attentive (Sleeping)"
                else:
                    label_text = "Not-Attentive (" + label + ")"
                color = (0, 0, 255)  # Red

            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # --- 5. NEW: Alert & Snapshot Logic (after all processing) ---

        if not is_frame_attentive:
            # Person is NOT attentive in this frame
            if non_attentive_start_time is None:
                # Just became non-attentive, start the timer
                non_attentive_start_time = time.time()
            else:
                # Timer is running, check elapsed time
                elapsed_time = time.time() - non_attentive_start_time

                # Display timer on screen
                timer_text = f"Not Attentive: {elapsed_time:.1f}s"
                cv2.putText(frame, timer_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                if elapsed_time > NON_ATTENTIVE_TIME_LIMIT and not alert_triggered:
                    # Time limit exceeded and we haven't triggered the alert yet
                    print(f"ALERT: User not attentive for {NON_ATTENTIVE_TIME_LIMIT} seconds.")

                    # 1. Save snapshot
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    snapshot_filename = os.path.join(SNAPSHOT_DIR, f"snapshot_{timestamp}.png")
                    cv2.imwrite(snapshot_filename, frame)
                    print(f"Saved snapshot: {snapshot_filename}")

                    # 2. Play sound in a new thread
                    alert_thread = threading.Thread(target=play_alert)
                    alert_thread.start()

                    # 3. Set flag so we don't re-trigger every frame
                    alert_triggered = True

        else:
            # Person IS attentive
            # Reset the timer and the trigger flag
            non_attentive_start_time = None
            alert_triggered = False

        # --- 6. Display the Results ---
        cv2.imshow('Student Attention Detector', frame)
        cv2.imshow('Face Emotion Probabilities using AI', canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"An error occurred: {e}")
        break

# --- 7. Cleanup ---
print("Exiting and cleaning up...")
cap.release()
cv2.destroyAllWindows()