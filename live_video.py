## fourth Improvement
# from keras.preprocessing.image import img_to_array
# import cv2
# import imutils
# from keras.models import load_model
# import numpy as np
#
# # --- NEW Imports ---
# import time
# import threading
# from datetime import datetime
# from playsound import playsound
# import os
# from collections import deque  # For the timeline graph
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
# # --- NEW: Alert Configuration ---
# ALERT_SOUND_FILE = "mixkit-urgent-simple-tone-loop-2976.wav"
# NON_ATTENTIVE_TIME_LIMIT = 5  # Seconds
# SNAPSHOT_DIR = "snapshots"
#
# # --- NEW: State variables for the timer ---
# non_attentive_start_time = None
# alert_triggered = False
#
# # --- NEW: Create snapshots directory ---
# os.makedirs(SNAPSHOT_DIR, exist_ok=True)
#
#
# # --- NEW: Function to play sound in a separate thread ---
# def play_alert():
#     try:
#         playsound(ALERT_SOUND_FILE)
#     except Exception as e:
#         print(f"Error playing sound: {e}")
#         print("Make sure 'alert.wav' (or .mp3) is in the folder and 'playsound' is installed.")
#
#
# # --- 2. Initialize Video Capture (Webcam Only) ---
#
# print("Starting webcam...")
# cap = cv2.VideoCapture(0)
#
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()
#
# # --- UI CHANGE: Renamed windows ---
# MAIN_WINDOW_NAME = 'Student Attention Monitor'
# DASHBOARD_WINDOW_NAME = 'Analysis Dashboard'
# cv2.namedWindow(MAIN_WINDOW_NAME)
# cv2.namedWindow(DASHBOARD_WINDOW_NAME)
#
# # --- MODIFIED --- Data stores for session summary (Idea 2 & 3)
# emotions_map = {
#     "angry": 0, "disgust": 0, "fear": 0, "happy": 0,
#     "sad": 0, "surprised": 0, "neutral": 0
# }
# non_attentive_alert_count = 0
# # --- MODIFIED --- Now stores (emotion_index, is_attentive) tuples
# emotion_history = deque(maxlen=380)
# EMOTIONS_IDX = {emotion: i for i, emotion in enumerate(EMOTIONS)}
#
# # --- 3. Start Real-Time Processing Loop ---
#
# while True:
#     try:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to grab frame from webcam.")
#             break
#
#         frame = imutils.resize(frame, width=400)
#         frame = cv2.flip(frame, 1)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#         # --- UI CHANGE: Create a semi-transparent header for status text ---
#         frame_h, frame_w, _ = frame.shape
#         overlay = frame.copy()
#         cv2.rectangle(overlay, (0, 0), (frame_w, 55), (0, 0, 0), -1)  # Header bar
#         alpha = 0.6  # Opacity
#         cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
#
#         # --- MODIFIED --- Increased canvas height for the new timeline graph
#         canvas = np.zeros((550, 400, 3), dtype="uint8")
#
#         # --- UI CHANGE: Add title to the Analysis canvas ---
#         cv2.putText(canvas, "Emotion Analysis", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
#                                               flags=cv2.CASCADE_SCALE_IMAGE)
#
#         is_frame_attentive = False  # Assume not attentive for this frame
#         current_emotion_index = -1  # -1 means no face
#         current_label = "---"
#
#         # --- 4. Process Detected Faces ---
#
#         if (len(faces) == 0):
#             # No face detected
#             is_frame_attentive = False
#             cv2.putText(frame, "Not-Attentive (student unavailable)", (10, 20),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#         else:
#             # Face was detected, process it
#             (x, y, w, h) = faces[0]  # Assume one face
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
#             roi_gray = gray[y:y + h, x:x + w]
#             roi_color = frame[y:y + h, x:x + w]
#
#             # --- Eye Detection (for Attentiveness) ---
#             eyes = eye_cascade.detectMultiScale(roi_gray)
#             for (ex, ey, ew, eh) in eyes[:2]:
#                 cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
#
#             # --- Determine Attentiveness ---
#             if (len(eyes) >= 1):
#                 is_frame_attentive = True  # Attentive if face AND eyes are found
#
#             # --- Emotion Prediction ---
#             roi_emotion = cv2.resize(roi_gray, (48, 48))
#             roi_emotion = roi_emotion.astype("float") / 255.0
#             roi_emotion = img_to_array(roi_emotion)
#             roi_emotion = np.expand_dims(roi_emotion, axis=0)
#
#             preds = emotion_classifier.predict(roi_emotion)[0]
#             current_emotion_index = np.argmax(preds)
#             current_label = EMOTIONS[current_emotion_index]
#             emotions_map[current_label] += 1
#
#             # --- Draw Probability Bars on Canvas ---
#             for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
#                 text = "{}: {:.2f}%".format(emotion, prob * 100)
#                 w = int(prob * 300)
#                 bar_y_start = (i * 35) + 50
#                 bar_y_end = (i * 35) + 80
#                 text_y = (i * 35) + 68
#                 cv2.rectangle(canvas, (7, bar_y_start),
#                               (w, bar_y_end), (255, 150, 50), -1)
#                 cv2.putText(canvas, text, (10, text_y),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#
#             # --- Set Label for Video Frame ---
#             if (is_frame_attentive):
#                 label_text = f"Attentive ({current_label})"
#                 color = (0, 255, 0)
#             else:
#                 if current_label == "neutral":
#                     label_text = "Not-Attentive (Sleeping)"
#                 else:
#                     label_text = f"Not-Attentive ({current_label})"
#                 color = (0, 0, 255)
#             cv2.putText(frame, label_text, (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#
#         # --- NEW --- Add data to timeline graph
#         # Store a tuple: (emotion_index, is_attentive)
#         emotion_history.append((current_emotion_index, is_frame_attentive))
#
#         # --- 5. Alert & Snapshot Logic ---
#
#         if not is_frame_attentive:
#             if non_attentive_start_time is None:
#                 non_attentive_start_time = time.time()
#             else:
#                 elapsed_time = time.time() - non_attentive_start_time
#                 timer_text = f"Not Attentive: {elapsed_time:.1f}s"
#                 cv2.putText(frame, timer_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#
#                 if elapsed_time > NON_ATTENTIVE_TIME_LIMIT and not alert_triggered:
#                     print(f"ALERT: User not attentive for {NON_ATTENTIVE_TIME_LIMIT} seconds.")
#                     non_attentive_alert_count += 1
#                     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#                     snapshot_filename = os.path.join(SNAPSHOT_DIR, f"snapshot_{timestamp}.png")
#                     cv2.imwrite(snapshot_filename, frame)
#                     print(f"Saved snapshot: {snapshot_filename}")
#                     alert_thread = threading.Thread(target=play_alert)
#                     alert_thread.start()
#                     alert_triggered = True
#         else:
#             # Person IS attentive
#             non_attentive_start_time = None
#             alert_triggered = False
#
#         # --- 6. Display the Results ---
#
#         # --- MODIFIED --- Draw the Emotion & Attentiveness Timeline Graph
#         graph_y_start = 330  # Y-position to start the graph
#         graph_y_end = 530
#         graph_height = 200  # Total height of graph area
#
#         cv2.putText(canvas, "Emotion & Attentiveness Timeline", (10, graph_y_start - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#         # Draw graph border
#         cv2.rectangle(canvas, (10, graph_y_start), (390, graph_y_end), (255, 255, 255), 1)
#
#         # --- NEW: Add Legend for Graph ---
#         cv2.rectangle(canvas, (15, graph_y_start + 10), (25, graph_y_start + 20), (0, 255, 0), -1)
#         cv2.putText(canvas, "Attentive", (30, graph_y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
#         cv2.rectangle(canvas, (115, graph_y_start + 10), (125, graph_y_start + 20), (0, 0, 255), -1)
#         cv2.putText(canvas, "Not-Attentive", (130, graph_y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
#                     1)
#
#         for i in range(1, len(emotion_history)):
#             # Get data for previous and current point
#             prev_emotion, prev_attentive = emotion_history[i - 1]
#             curr_emotion, curr_attentive = emotion_history[i]
#
#             # Don't draw if there was a gap (no face)
#             if prev_emotion == -1 or curr_emotion == -1:
#                 continue
#
#             # Y-value is emotion index (0-6).
#             y1_px = int(graph_y_end - (prev_emotion / 6.0) * graph_height)
#             y2_px = int(graph_y_end - (curr_emotion / 6.0) * graph_height)
#
#             # X-value is just the position in the list
#             x_px = 10 + i
#
#             # --- MODIFIED --- Color is based on attentiveness
#             color = (0, 255, 0) if curr_attentive else (0, 0, 255)
#
#             cv2.line(canvas, (x_px - 1, y1_px), (x_px, y2_px), color, 1)
#         # --- End of Timeline Graph ---
#
#         cv2.imshow(MAIN_WINDOW_NAME, frame)
#         cv2.imshow(DASHBOARD_WINDOW_NAME, canvas)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         break
#
# # --- 7. Cleanup ---
# print("Exiting and cleaning up...")
# cap.release()
# # --- MODIFIED --- Destroy specific windows before summary
# cv2.destroyWindow(MAIN_WINDOW_NAME)
# cv2.destroyWindow(DASHBOARD_WINDOW_NAME)
#
# # --- 8. NEW: Show & Save Session Summary Report ---
#
# print("Generating session summary...")
# summary_canvas = np.zeros((400, 500, 3), dtype="uint8")
# summary_canvas.fill(255)  # White background
#
# # --- Title ---
# cv2.putText(summary_canvas, "Session Summary", (10, 40),
#             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
#
# # --- Alert Count ---
# cv2.putText(summary_canvas, f"Total Non-Attentive Alerts: {non_attentive_alert_count}", (10, 90),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
#
# # --- Emotion Bar Graph ---
# cv2.putText(summary_canvas, "Emotion Analysis:", (10, 150),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
#
# y_pos = 180
# total_emotions = sum(emotions_map.values())
#
# if total_emotions > 0:
#     for emotion, count in emotions_map.items():
#         percentage = count / total_emotions
#         text = f"{emotion.capitalize()}:"
#         cv2.putText(summary_canvas, text, (10, y_pos),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
#
#         bar_width_pixels = int(percentage * 350)
#         cv2.rectangle(summary_canvas, (100, y_pos - 15), (100 + bar_width_pixels, y_pos + 5),
#                       (255, 150, 50), -1)
#
#         cv2.putText(summary_canvas, f"{(percentage * 100):.1f}%", (105 + bar_width_pixels, y_pos),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
#
#         y_pos += 30
# else:
#     cv2.putText(summary_canvas, "No emotion data was collected.", (10, 180),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
#
# # --- Final Instructions ---
# cv2.putText(summary_canvas, "Press any key to exit.", (10, 380),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
#
# # --- NEW: Save the summary canvas to a file ---
# try:
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     summary_filename = os.path.join(SNAPSHOT_DIR, f"summary_{timestamp}.png")
#     cv2.imwrite(summary_filename, summary_canvas)
#     print(f"Successfully saved summary report to: {summary_filename}")
# except Exception as e:
#     print(f"Error: Failed to save summary report. {e}")
#
# # --- Show the summary window ---
# cv2.imshow("Session Summary", summary_canvas)
# cv2.waitKey(0)  # Wait indefinitely until a key is pressed
# cv2.destroyAllWindows()  # Final cleanup


# from keras.preprocessing.image import img_to_array
# import cv2
# import imutils
# from keras.models import load_model
# import numpy as np
#
# # --- NEW Imports ---
# import time
# import threading
# from datetime import datetime
# from playsound import playsound
# import os
# from collections import deque  # For the timeline graph
#
# # --- 1. Load Models & Configuration ---
# # (All code from section 1 is unchanged)
# face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_eye.xml')
# video_emotion_model_path = 'model_num.hdf5'
# try:
#     emotion_classifier = load_model(video_emotion_model_path, compile=False)
# except IOError:
#     print(f"Error: Model file not found at {video_emotion_model_path}")
#     print("Please make sure 'model_num.hdf5' is in the same folder as this script.")
#     exit()
# EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]
# ALERT_SOUND_FILE = "mixkit-urgent-simple-tone-loop-2976.wav"
# NON_ATTENTIVE_TIME_LIMIT = 5
# SNAPSHOT_DIR = "snapshots"
# non_attentive_start_time = None
# alert_triggered = False
# os.makedirs(SNAPSHOT_DIR, exist_ok=True)
#
#
# def play_alert():
#     try:
#         playsound(ALERT_SOUND_FILE)
#     except Exception as e:
#         print(f"Error playing sound: {e}")
#         print("Make sure 'alert.wav' (or .mp3) is in the folder and 'playsound' is installed.")
#
#
# # --- 2. Initialize Video Capture (Webcam Only) ---
# print("Starting webcam...")
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()
# MAIN_WINDOW_NAME = 'Student Attention Monitor'
# DASHBOARD_WINDOW_NAME = 'Analysis Dashboard'
# cv2.namedWindow(MAIN_WINDOW_NAME)
# cv2.namedWindow(DASHBOARD_WINDOW_NAME)
# emotions_map = {"angry": 0, "disgust": 0, "fear": 0, "happy": 0, "sad": 0, "surprised": 0, "neutral": 0}
# non_attentive_alert_count = 0
# emotion_history = deque(maxlen=380)
# EMOTIONS_IDX = {emotion: i for i, emotion in enumerate(EMOTIONS)}
#
# # --- 3. Start Real-Time Processing Loop ---
#
# # --- NEW --- This outer 'try' block will catch 'Ctrl+C'
# try:
#     while True:
#         # --- This inner 'try' block catches frame-level errors ---
#         try:
#             ret, frame = cap.read()
#             if not ret:
#                 print("Error: Failed to grab frame from webcam.")
#                 break
#
#             frame = imutils.resize(frame, width=400)
#             frame = cv2.flip(frame, 1)
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#             frame_h, frame_w, _ = frame.shape
#             overlay = frame.copy()
#             cv2.rectangle(overlay, (0, 0), (frame_w, 55), (0, 0, 0), -1)
#             alpha = 0.6
#             cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
#
#             canvas = np.zeros((550, 400, 3), dtype="uint8")
#             cv2.putText(canvas, "Emotion Analysis", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#
#             faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
#                                                   flags=cv2.CASCADE_SCALE_IMAGE)
#
#             is_frame_attentive = False
#             current_emotion_index = -1
#             current_label = "---"
#
#             # --- 4. Process Detected Faces ---
#             if (len(faces) == 0):
#                 is_frame_attentive = False
#                 cv2.putText(frame, "Not-Attentive (student unavailable)", (10, 20),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#             else:
#                 (x, y, w, h) = faces[0]
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
#                 roi_gray = gray[y:y + h, x:x + w]
#                 roi_color = frame[y:y + h, x:x + w]
#
#                 eyes = eye_cascade.detectMultiScale(roi_gray)
#                 for (ex, ey, ew, eh) in eyes[:2]:
#                     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
#
#                 if (len(eyes) >= 1):
#                     is_frame_attentive = True
#
#                 roi_emotion = cv2.resize(roi_gray, (48, 48))
#                 roi_emotion = roi_emotion.astype("float") / 255.0
#                 roi_emotion = img_to_array(roi_emotion)
#                 roi_emotion = np.expand_dims(roi_emotion, axis=0)
#
#                 preds = emotion_classifier.predict(roi_emotion)[0]
#                 current_emotion_index = np.argmax(preds)
#                 current_label = EMOTIONS[current_emotion_index]
#                 emotions_map[current_label] += 1
#
#                 for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
#                     text = "{}: {:.2f}%".format(emotion, prob * 100)
#                     w = int(prob * 300)
#                     bar_y_start = (i * 35) + 50
#                     bar_y_end = (i * 35) + 80
#                     text_y = (i * 35) + 68
#                     cv2.rectangle(canvas, (7, bar_y_start),
#                                   (w, bar_y_end), (255, 150, 50), -1)
#                     cv2.putText(canvas, text, (10, text_y),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#
#                 if (is_frame_attentive):
#                     label_text = f"Attentive ({current_label})"
#                     color = (0, 255, 0)
#                 else:
#                     if current_label == "neutral":
#                         label_text = "Not-Attentive (Sleeping)"
#                     else:
#                         label_text = f"Not-Attentive ({current_label})"
#                     color = (0, 0, 255)
#                 cv2.putText(frame, label_text, (x, y - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#
#             emotion_history.append((current_emotion_index, is_frame_attentive))
#
#             # --- 5. Alert & Snapshot Logic ---
#             if not is_frame_attentive:
#                 if non_attentive_start_time is None:
#                     non_attentive_start_time = time.time()
#                 else:
#                     elapsed_time = time.time() - non_attentive_start_time
#                     timer_text = f"Not Attentive: {elapsed_time:.1f}s"
#                     cv2.putText(frame, timer_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#
#                     if elapsed_time > NON_ATTENTIVE_TIME_LIMIT and not alert_triggered:
#                         print(f"ALERT: User not attentive for {NON_ATTENTIVE_TIME_LIMIT} seconds.")
#                         non_attentive_alert_count += 1
#                         timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#                         snapshot_filename = os.path.join(SNAPSHOT_DIR, f"snapshot_{timestamp}.png")
#                         cv2.imwrite(snapshot_filename, frame)
#                         print(f"Saved snapshot: {snapshot_filename}")
#                         alert_thread = threading.Thread(target=play_alert)
#                         alert_thread.start()
#                         alert_triggered = True
#             else:
#                 non_attentive_start_time = None
#                 alert_triggered = False
#
#             # --- 6. Display the Results ---
#             graph_y_start = 330
#             graph_y_end = 530
#             graph_height = 200
#
#             cv2.putText(canvas, "Emotion & Attentiveness Timeline", (10, graph_y_start - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#             cv2.rectangle(canvas, (10, graph_y_start), (390, graph_y_end), (255, 255, 255), 1)
#             cv2.rectangle(canvas, (15, graph_y_start + 10), (25, graph_y_start + 20), (0, 255, 0), -1)
#             cv2.putText(canvas, "Attentive", (30, graph_y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
#                         1)
#             cv2.rectangle(canvas, (115, graph_y_start + 10), (125, graph_y_start + 20), (0, 0, 255), -1)
#             cv2.putText(canvas, "Not-Attentive", (130, graph_y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
#                         (255, 255, 255), 1)
#
#             for i in range(1, len(emotion_history)):
#                 prev_emotion, prev_attentive = emotion_history[i - 1]
#                 curr_emotion, curr_attentive = emotion_history[i]
#                 if prev_emotion == -1 or curr_emotion == -1:
#                     continue
#                 y1_px = int(graph_y_end - (prev_emotion / 6.0) * graph_height)
#                 y2_px = int(graph_y_end - (curr_emotion / 6.0) * graph_height)
#                 x_px = 10 + i
#                 color = (0, 255, 0) if curr_attentive else (0, 0, 255)
#                 cv2.line(canvas, (x_px - 1, y1_px), (x_px, y2_px), color, 1)
#
#             cv2.imshow(MAIN_WINDOW_NAME, frame)
#             cv2.imshow(DASHBOARD_WINDOW_NAME, canvas)
#
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#         except Exception as e:
#             # This catches errors *within* a frame, so the loop can continue
#             print(f"An error occurred in the loop: {e}")
#             break  # Break from the loop on an internal error
#
# # --- NEW --- This 'finally' block will run NO MATTER WHAT
# finally:
#     # --- 7. Cleanup ---
#     print("Exiting and cleaning up...")
#     cap.release()
#     cv2.destroyWindow(MAIN_WINDOW_NAME)
#     cv2.destroyWindow(DASHBOARD_WINDOW_NAME)
#
#     # --- 8. NEW: Show & Save Session Summary Report ---
#     print("Generating session summary...")
#     summary_canvas = np.zeros((400, 500, 3), dtype="uint8")
#     summary_canvas.fill(255)  # White background
#
#     # --- Title ---
#     cv2.putText(summary_canvas, "Session Summary", (10, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
#
#     # --- Alert Count ---
#     cv2.putText(summary_canvas, f"Total Non-Attentive Alerts: {non_attentive_alert_count}", (10, 90),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
#
#     # --- Emotion Bar Graph ---
#     cv2.putText(summary_canvas, "Emotion Analysis:", (10, 150),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
#     y_pos = 180
#     total_emotions = sum(emotions_map.values())
#
#     if total_emotions > 0:
#         for emotion, count in emotions_map.items():
#             percentage = count / total_emotions
#             text = f"{emotion.capitalize()}:"
#             cv2.putText(summary_canvas, text, (10, y_pos),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
#             bar_width_pixels = int(percentage * 350)
#             cv2.rectangle(summary_canvas, (100, y_pos - 15), (100 + bar_width_pixels, y_pos + 5),
#                           (255, 150, 50), -1)
#             cv2.putText(summary_canvas, f"{(percentage * 100):.1f}%", (105 + bar_width_pixels, y_pos),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
#             y_pos += 30
#     else:
#         cv2.putText(summary_canvas, "No emotion data was collected.", (10, 180),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
#
#     # --- Final Instructions ---
#     cv2.putText(summary_canvas, "Press any key to exit.", (10, 380),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
#
#     # --- NEW: Save the summary canvas to a file ---
#     try:
#         timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         summary_filename = os.path.join(SNAPSHOT_DIR, f"summary_{timestamp}.png")
#         cv2.imwrite(summary_filename, summary_canvas)
#         print(f"Successfully saved summary report to: {summary_filename}")
#     except Exception as e:
#         print(f"Error: Failed to save summary report. {e}")
#
#     # --- Show the summary window ---
#     cv2.imshow("Session Summary", summary_canvas)
#     cv2.waitKey(0)  # Wait indefinitely until a key is pressed
#     cv2.destroyAllWindows()  # Final cleanup

# from keras.preprocessing.image import img_to_array
# import cv2
# import imutils
# from keras.models import load_model
# import numpy as np
#
# # --- NEW Imports ---
# import time
# import threading
# from datetime import datetime
# from playsound import playsound
# import os
# from collections import deque  # For the timeline graph
#
# # --- 1. Load Models & Configuration ---
# # (All code from section 1 is unchanged)
# face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_eye.xml')
# video_emotion_model_path = 'model_num.hdf5'
# try:
#     emotion_classifier = load_model(video_emotion_model_path, compile=False)
# except IOError:
#     print(f"Error: Model file not found at {video_emotion_model_path}")
#     print("Please make sure 'model_num.hdf5' is in the same folder as this script.")
#     exit()
# EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]
# ALERT_SOUND_FILE = "mixkit-urgent-simple-tone-loop-2976.wav"
# NON_ATTENTIVE_TIME_LIMIT = 5
# SNAPSHOT_DIR = "snapshots"
# non_attentive_start_time = None
# alert_triggered = False
# os.makedirs(SNAPSHOT_DIR, exist_ok=True)
#
#
# def play_alert():
#     try:
#         playsound(ALERT_SOUND_FILE)
#     except Exception as e:
#         print(f"Error playing sound: {e}")
#         print("Make sure 'alert.wav' (or .mp3) is in the folder and 'playsound' is installed.")
#
#
# # --- 2. Initialize Video Capture (Webcam Only) ---
# print("Starting webcam...")
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()
# MAIN_WINDOW_NAME = 'Student Attention Monitor'
# DASHBOARD_WINDOW_NAME = 'Analysis Dashboard'
# cv2.namedWindow(MAIN_WINDOW_NAME)
# cv2.namedWindow(DASHBOARD_WINDOW_NAME)
# emotions_map = {"angry": 0, "disgust": 0, "fear": 0, "happy": 0, "sad": 0, "surprised": 0, "neutral": 0}
# non_attentive_alert_count = 0
# emotion_history = deque(maxlen=380)  # For real-time graph
# EMOTIONS_IDX = {emotion: i for i, emotion in enumerate(EMOTIONS)}
#
# # --- NEW --- List to store the *entire* session's history
# full_session_history = []
#
# # --- 3. Start Real-Time Processing Loop ---
# try:
#     while True:
#         try:
#             ret, frame = cap.read()
#             if not ret:
#                 print("Error: Failed to grab frame from webcam.")
#                 break
#
#             frame = imutils.resize(frame, width=400)
#             frame = cv2.flip(frame, 1)
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#             frame_h, frame_w, _ = frame.shape
#             overlay = frame.copy()
#             cv2.rectangle(overlay, (0, 0), (frame_w, 55), (0, 0, 0), -1)
#             alpha = 0.6
#             cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
#
#             canvas = np.zeros((550, 400, 3), dtype="uint8")
#             cv2.putText(canvas, "Emotion Analysis", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#
#             faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
#                                                   flags=cv2.CASCADE_SCALE_IMAGE)
#
#             is_frame_attentive = False
#             current_emotion_index = -1
#             current_label = "---"
#
#             # --- 4. Process Detected Faces ---
#             if (len(faces) == 0):
#                 is_frame_attentive = False
#                 cv2.putText(frame, "Not-Attentive (student unavailable)", (10, 20),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#             else:
#                 (x, y, w, h) = faces[0]
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
#                 roi_gray = gray[y:y + h, x:x + w]
#                 roi_color = frame[y:y + h, x:x + w]
#
#                 eyes = eye_cascade.detectMultiScale(roi_gray)
#                 for (ex, ey, ew, eh) in eyes[:2]:
#                     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
#
#                 if (len(eyes) >= 1):
#                     is_frame_attentive = True
#
#                 roi_emotion = cv2.resize(roi_gray, (48, 48))
#                 roi_emotion = roi_emotion.astype("float") / 255.0
#                 roi_emotion = img_to_array(roi_emotion)
#                 roi_emotion = np.expand_dims(roi_emotion, axis=0)
#
#                 preds = emotion_classifier.predict(roi_emotion)[0]
#                 current_emotion_index = np.argmax(preds)
#                 current_label = EMOTIONS[current_emotion_index]
#                 emotions_map[current_label] += 1
#
#                 for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
#                     text = "{}: {:.2f}%".format(emotion, prob * 100)
#                     w = int(prob * 300)
#                     bar_y_start = (i * 35) + 50
#                     bar_y_end = (i * 35) + 80
#                     text_y = (i * 35) + 68
#                     cv2.rectangle(canvas, (7, bar_y_start),
#                                   (w, bar_y_end), (255, 150, 50), -1)
#                     cv2.putText(canvas, text, (10, text_y),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#
#                 if (is_frame_attentive):
#                     label_text = f"Attentive ({current_label})"
#                     color = (0, 255, 0)
#                 else:
#                     if current_label == "neutral":
#                         label_text = "Not-Attentive (Sleeping)"
#                     else:
#                         label_text = f"Not-Attentive ({current_label})"
#                     color = (0, 0, 255)
#                 cv2.putText(frame, label_text, (x, y - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#
#             # --- MODIFIED --- Add data to BOTH history lists
#             data_point = (current_emotion_index, is_frame_attentive)
#             emotion_history.append(data_point)  # For real-time
#             full_session_history.append(data_point)  # For summary
#
#             # --- 5. Alert & Snapshot Logic ---
#             if not is_frame_attentive:
#                 if non_attentive_start_time is None:
#                     non_attentive_start_time = time.time()
#                 else:
#                     elapsed_time = time.time() - non_attentive_start_time
#                     timer_text = f"Not Attentive: {elapsed_time:.1f}s"
#                     cv2.putText(frame, timer_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#
#                     if elapsed_time > NON_ATTENTIVE_TIME_LIMIT and not alert_triggered:
#                         print(f"ALERT: User not attentive for {NON_ATTENTIVE_TIME_LIMIT} seconds.")
#                         non_attentive_alert_count += 1
#                         timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#                         snapshot_filename = os.path.join(SNAPSHOT_DIR, f"snapshot_{timestamp}.png")
#                         cv2.imwrite(snapshot_filename, frame)
#                         print(f"Saved snapshot: {snapshot_filename}")
#                         alert_thread = threading.Thread(target=play_alert)
#                         alert_thread.start()
#                         alert_triggered = True
#             else:
#                 non_attentive_start_time = None
#                 alert_triggered = False
#
#             # --- 6. Display the Results (Real-time Graph) ---
#             graph_y_start = 330
#             graph_y_end = 530
#             graph_height = 200
#
#             cv2.putText(canvas, "Emotion & Attentiveness Timeline", (10, graph_y_start - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#             cv2.rectangle(canvas, (10, graph_y_start), (390, graph_y_end), (255, 255, 255), 1)
#             cv2.rectangle(canvas, (15, graph_y_start + 10), (25, graph_y_start + 20), (0, 255, 0), -1)
#             cv2.putText(canvas, "Attentive", (30, graph_y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
#                         1)
#             cv2.rectangle(canvas, (115, graph_y_start + 10), (125, graph_y_start + 20), (0, 0, 255), -1)
#             cv2.putText(canvas, "Not-Attentive", (130, graph_y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
#                         (255, 255, 255), 1)
#
#             for i in range(1, len(emotion_history)):
#                 prev_emotion, prev_attentive = emotion_history[i - 1]
#                 curr_emotion, curr_attentive = emotion_history[i]
#                 if prev_emotion == -1 or curr_emotion == -1:
#                     continue
#                 y1_px = int(graph_y_end - (prev_emotion / 6.0) * graph_height)
#                 y2_px = int(graph_y_end - (curr_emotion / 6.0) * graph_height)
#                 x_px = 10 + i
#                 color = (0, 255, 0) if curr_attentive else (0, 0, 255)
#                 cv2.line(canvas, (x_px - 1, y1_px), (x_px, y2_px), color, 1)
#
#             cv2.imshow(MAIN_WINDOW_NAME, frame)
#             cv2.imshow(DASHBOARD_WINDOW_NAME, canvas)
#
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#         except Exception as e:
#             print(f"An error occurred in the loop: {e}")
#             break
#
# # --- NEW --- This 'finally' block will run NO MATTER WHAT
# finally:
#     # --- 7. Cleanup ---
#     print("Exiting and cleaning up...")
#     cap.release()
#     cv2.destroyWindow(MAIN_WINDOW_NAME)
#     cv2.destroyWindow(DASHBOARD_WINDOW_NAME)
#
#     # --- 8. NEW: Show & Save Session Summary Report ---
#     print("Generating session summary...")
#
#     # --- MODIFIED --- Increased canvas height to 600px for the new graph
#     summary_canvas = np.zeros((600, 500, 3), dtype="uint8")
#     summary_canvas.fill(255)  # White background
#
#     # --- Title ---
#     cv2.putText(summary_canvas, "Session Summary", (10, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
#
#     # --- Alert Count ---
#     cv2.putText(summary_canvas, f"Total Non-Attentive Alerts: {non_attentive_alert_count}", (10, 90),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
#
#     # --- Emotion Bar Graph ---
#     cv2.putText(summary_canvas, "Emotion Analysis:", (10, 150),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
#     y_pos = 180
#     total_emotions = sum(emotions_map.values())
#
#     if total_emotions > 0:
#         for emotion, count in emotions_map.items():
#             percentage = count / total_emotions
#             text = f"{emotion.capitalize()}:"
#             cv2.putText(summary_canvas, text, (10, y_pos),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
#             bar_width_pixels = int(percentage * 350)
#             cv2.rectangle(summary_canvas, (100, y_pos - 15), (100 + bar_width_pixels, y_pos + 5),
#                           (255, 150, 50), -1)
#             cv2.putText(summary_canvas, f"{(percentage * 100):.1f}%", (105 + bar_width_pixels, y_pos),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
#             y_pos += 30
#     else:
#         cv2.putText(summary_canvas, "No emotion data was collected.", (10, 180),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
#
#     # --- NEW: Full Session Timeline Graph ---
#     graph_area_y_start = y_pos + 20  # Start graph below the bar chart
#     graph_area_height = 150
#     graph_area_y_end = graph_area_y_start + graph_area_height
#     graph_area_x_start = 10
#     graph_area_x_end = 490
#     graph_area_width = graph_area_x_end - graph_area_x_start
#
#     cv2.putText(summary_canvas, "Full Session Timeline:", (10, graph_area_y_start - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
#     cv2.rectangle(summary_canvas, (graph_area_x_start, graph_area_y_start), (graph_area_x_end, graph_area_y_end),
#                   (0, 0, 0), 1)
#
#     if len(full_session_history) > 1:
#         # Draw the graph line by line
#         for i in range(1, len(full_session_history)):
#             # Get data for previous and current point
#             prev_emotion, prev_attentive = full_session_history[i - 1]
#             curr_emotion, curr_attentive = full_session_history[i]
#
#             # Don't draw if there was a gap (no face)
#             if prev_emotion == -1 or curr_emotion == -1:
#                 continue
#
#             # --- Scale X and Y to fit the graph area ---
#             # Y-value (Emotion)
#             y1_px = int(graph_area_y_end - (prev_emotion / 6.0) * graph_area_height)
#             y2_px = int(graph_area_y_end - (curr_emotion / 6.0) * graph_area_height)
#
#             # X-value (Time)
#             x1_px = int(graph_area_x_start + ((i - 1) / len(full_session_history)) * graph_area_width)
#             x2_px = int(graph_area_x_start + (i / len(full_session_history)) * graph_area_width)
#
#             # Color (Attentiveness)
#             color = (0, 200, 0) if curr_attentive else (0, 0, 255)  # Green/Red
#
#             cv2.line(summary_canvas, (x1_px, y1_px), (x2_px, y2_px), color, 1)
#     else:
#         cv2.putText(summary_canvas, "Not enough data for timeline.", (10, graph_area_y_start + 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
#
#     # --- Legend for new graph ---
#     cv2.rectangle(summary_canvas, (15, graph_area_y_end + 10), (25, graph_area_y_end + 20), (0, 200, 0), -1)
#     cv2.putText(summary_canvas, "Attentive", (30, graph_area_y_end + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
#     cv2.rectangle(summary_canvas, (115, graph_area_y_end + 10), (125, graph_area_y_end + 20), (0, 0, 255), -1)
#     cv2.putText(summary_canvas, "Not-Attentive", (130, graph_area_y_end + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
#                 1)
#
#     # --- Final Instructions ---
#     cv2.putText(summary_canvas, "Press any key to exit.", (10, 580),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
#
#     # --- Save the summary canvas to a file ---
#     try:
#         timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         summary_filename = os.path.join(SNAPSHOT_DIR, f"summary_{timestamp}.png")
#         cv2.imwrite(summary_filename, summary_canvas)
#         print(f"Successfully saved summary report to: {summary_filename}")
#     except Exception as e:
#         print(f"Error: Failed to save summary report. {e}")
#
#     # --- Show the summary window ---
#     cv2.imshow("Session Summary", summary_canvas)
#     cv2.waitKey(0)  # Wait indefinitely until a key is pressed
#     cv2.destroyAllWindows()  # Final cleanup




## fifth Improvement

# from keras.preprocessing.image import img_to_array
# import cv2
# import imutils
# from keras.models import load_model
# import numpy as np
#
# # --- NEW Imports ---
# import time
# import threading
# from datetime import datetime
# from playsound import playsound
# import os
# from collections import deque  # For the timeline graph
#
# # --- 1. Load Models & Configuration ---
# # (All code from section 1 is unchanged)
# face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_eye.xml')
# video_emotion_model_path = 'model_num.hdf5'
# try:
#     emotion_classifier = load_model(video_emotion_model_path, compile=False)
# except IOError:
#     print(f"Error: Model file not found at {video_emotion_model_path}")
#     print("Please make sure 'model_num.hdf5' is in the same folder as this script.")
#     exit()
# EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]
# ALERT_SOUND_FILE = "mixkit-urgent-simple-tone-loop-2976.wav"
# NON_ATTENTIVE_TIME_LIMIT = 5
# SNAPSHOT_DIR = "snapshots"
# non_attentive_start_time = None
# alert_triggered = False
# os.makedirs(SNAPSHOT_DIR, exist_ok=True)
#
#
# def play_alert():
#     try:
#         playsound(ALERT_SOUND_FILE)
#     except Exception as e:
#         print(f"Error playing sound: {e}")
#         print("Make sure 'alert.wav' (or .mp3) is in the folder and 'playsound' is installed.")

from keras.preprocessing.image import img_to_array
import cv2
import imutils
from keras.models import load_model
import numpy as np

# --- NEW Imports ---
import time
import threading
from datetime import datetime
import os
from collections import deque
# --- NEW: Import pygame for stable sound ---
import pygame

# --- 1. Load Models & Configuration ---
face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_eye.xml')
video_emotion_model_path = 'model_num.hdf5'
try:
    emotion_classifier = load_model(video_emotion_model_path, compile=False)
except IOError:
    print(f"Error: Model file not found at {video_emotion_model_path}")
    print("Please make sure 'model_num.hdf5' is in the same folder as this script.")
    exit()
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]
ALERT_SOUND_FILE = "mixkit-urgent-simple-tone-loop-2976.wav"
NON_ATTENTIVE_TIME_LIMIT =3
SNAPSHOT_DIR = "snapshots"
non_attentive_start_time = None
alert_triggered = False
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# --- NEW: Initialize Pygame Mixer and Load Sound ---
# This is much more reliable than playsound
try:
    pygame.mixer.init()
    alert_sound = pygame.mixer.Sound(ALERT_SOUND_FILE)
except Exception as e:
    print(f"Error initializing sound: {e}")
    print(f"Please make sure 'pygame' is installed (pip install pygame)")
    print(f"And that {ALERT_SOUND_FILE} is in the same folder as the script.")
    alert_sound = None

# --- MODIFIED: play_alert function ---
def play_alert():
    if alert_sound:
        try:
            alert_sound.play()
        except Exception as e:
            print(f"Error playing sound: {e}")
    else:
        print("Sound not initialized, cannot play alert.")
def get_session_tag(total_attentive_time, total_non_attentive_time, non_attentive_alert_count, emotions_map):
    total_session_time = total_attentive_time + total_non_attentive_time

    if total_session_time < 10:  # Not enough data for a meaningful tag
        return "Short Session"

    attentive_percentage = total_attentive_time / total_session_time
    session_tag = "Mixed Focus"  # Default

    if attentive_percentage >= 0.85:
        session_tag = "Highly Focused"
    elif attentive_percentage >= 0.65:
        session_tag = "Generally Attentive"
    elif attentive_percentage <= 0.35:
        session_tag = "Highly Distracted"
    elif non_attentive_alert_count >= 5:
        session_tag = "Frequently Inattentive"

    # Emotional override
    total_emotions = sum(emotions_map.values())
    if total_emotions > 0:
        sad_angry_fear_pct = (emotions_map.get("sad", 0) + emotions_map.get("angry", 0) + emotions_map.get("fear",
                                                                                                           0)) / total_emotions
        if sad_angry_fear_pct > 0.4 and attentive_percentage < 0.6:
            session_tag = "Emotionally Distracted"

    return session_tag


# --- 2. Initialize Video Capture (Webcam Only) ---
print("Starting webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
MAIN_WINDOW_NAME = 'Student Attention Monitor'
DASHBOARD_WINDOW_NAME = 'Analysis Dashboard'
cv2.namedWindow(MAIN_WINDOW_NAME)
cv2.namedWindow(DASHBOARD_WINDOW_NAME)
emotions_map = {"angry": 0, "disgust": 0, "fear": 0, "happy": 0, "sad": 0, "surprised": 0, "neutral": 0}
non_attentive_alert_count = 0

# --- MODIFIED --- History now stores (timestamp, emotion_idx, is_attentive, non_attentive_duration)
emotion_history = deque(maxlen=380)  # For real-time graph
EMOTIONS_IDX = {emotion: i for i, emotion in enumerate(EMOTIONS)}

# --- NEW --- List to store the *entire* session's history
full_session_history = []

# --- 3. Start Real-Time Processing Loop ---
try:
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame from webcam.")
                break

            frame = imutils.resize(frame, width=400)
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame_h, frame_w, _ = frame.shape
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame_w, 55), (0, 0, 0), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            canvas = np.zeros((550, 400, 3), dtype="uint8")
            cv2.putText(canvas, "Emotion Analysis", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)

            is_frame_attentive = False
            current_emotion_index = -1
            current_label = "---"

            # --- 4. Process Detected Faces ---
            if (len(faces) == 0):
                is_frame_attentive = False
                cv2.putText(frame, "Not-Attentive (student unavailable)", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes[:2]:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                if (len(eyes) > 1):
                    is_frame_attentive = True

                roi_emotion = cv2.resize(roi_gray, (48, 48))
                roi_emotion = roi_emotion.astype("float") / 255.0
                roi_emotion = img_to_array(roi_emotion)
                roi_emotion = np.expand_dims(roi_emotion, axis=0)

                preds = emotion_classifier.predict(roi_emotion)[0]
                current_emotion_index = np.argmax(preds)
                current_label = EMOTIONS[current_emotion_index]
                emotions_map[current_label] += 1

                for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                    text = "{}: {:.2f}%".format(emotion, prob * 100)
                    w = int(prob * 300)
                    bar_y_start = (i * 35) + 50
                    bar_y_end = (i * 35) + 80
                    text_y = (i * 35) + 68
                    cv2.rectangle(canvas, (7, bar_y_start),
                                  (w, bar_y_end), (255, 150, 50), -1)
                    cv2.putText(canvas, text, (10, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if (is_frame_attentive):
                    label_text = f"Attentive ({current_label})"
                    color = (0, 255, 0)
                else:
                    if current_label == "neutral":
                        label_text = "Not-Attentive (Sleeping)"
                    else:
                        label_text = f"Not-Attentive ({current_label})"
                    color = (0, 0, 255)
                cv2.putText(frame, label_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- 5. MODIFIED: Alert & Snapshot Logic (now calculates duration) ---
            current_non_attentive_duration = 0.0

            if not is_frame_attentive:
                if non_attentive_start_time is None:
                    non_attentive_start_time = time.time()
                else:
                    elapsed_time = time.time() - non_attentive_start_time
                    current_non_attentive_duration = elapsed_time  # Store this
                    timer_text = f"Not Attentive: {elapsed_time:.1f}s"
                    cv2.putText(frame, timer_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    if elapsed_time > NON_ATTENTIVE_TIME_LIMIT and not alert_triggered:
                        print(f"ALERT: User not attentive for {NON_ATTENTIVE_TIME_LIMIT} seconds.")
                        non_attentive_alert_count += 1
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        snapshot_filename = os.path.join(SNAPSHOT_DIR, f"snapshot_{timestamp}.png")
                        cv2.imwrite(snapshot_filename, frame)
                        print(f"Saved snapshot: {snapshot_filename}")
                        alert_thread = threading.Thread(target=play_alert)
                        alert_thread.start()
                        alert_triggered = True
            else:
                non_attentive_start_time = None
                alert_triggered = False
                current_non_attentive_duration = 0.0

            # --- MODIFIED --- Add comprehensive data point to BOTH history lists
            current_timestamp = time.time()
            data_point = (current_timestamp, current_emotion_index, is_frame_attentive, current_non_attentive_duration)
            emotion_history.append(data_point)  # For real-time
            full_session_history.append(data_point)  # For summary

            # --- 6. Display the Results (Real-time Graph) ---
            graph_y_start = 330
            graph_y_end = 530
            graph_height = 200

            cv2.putText(canvas, "Emotion & Attentiveness Timeline", (10, graph_y_start - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.rectangle(canvas, (10, graph_y_start), (390, graph_y_end), (255, 255, 255), 1)

            # --- MODIFIED --- Legend updated with "Alerted"
            cv2.rectangle(canvas, (15, graph_y_start + 10), (25, graph_y_start + 20), (0, 255, 0), -1)
            cv2.putText(canvas, "Attentive", (30, graph_y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
                        1)
            cv2.rectangle(canvas, (115, graph_y_start + 10), (125, graph_y_start + 20), (0, 0, 255), -1)
            cv2.putText(canvas, "Not-Attentive", (130, graph_y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1)
            cv2.rectangle(canvas, (235, graph_y_start + 10), (245, graph_y_start + 20), (0, 0, 0), -1)
            cv2.putText(canvas, "Alerted", (250, graph_y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            for i in range(1, len(emotion_history)):
                # --- MODIFIED --- Unpack new data point structure
                _, prev_emotion, _, _ = emotion_history[i - 1]
                _, curr_emotion, curr_attentive, curr_duration = emotion_history[i]

                if prev_emotion == -1 or curr_emotion == -1:
                    continue

                y1_px = int(graph_y_end - (prev_emotion / 6.0) * graph_height)
                y2_px = int(graph_y_end - (curr_emotion / 6.0) * graph_height)
                x_px = 10 + i

                # --- MODIFIED --- Color logic now includes BLACK for alerted state
                if curr_attentive:
                    color = (0, 255, 0)  # Green
                else:
                    if curr_duration > NON_ATTENTIVE_TIME_LIMIT:
                        color = (0, 0, 0)  # Black
                    else:
                        color = (0, 0, 255)  # Red

                cv2.line(canvas, (x_px - 1, y1_px), (x_px, y2_px), color, 1)

            cv2.imshow(MAIN_WINDOW_NAME, frame)
            cv2.imshow(DASHBOARD_WINDOW_NAME, canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"An error occurred in the loop: {e}")
            break

# --- NEW --- This 'finally' block will run NO MATTER WHAT
finally:
    # --- 7. Cleanup ---
    print("Exiting and cleaning up...")
    cap.release()
    cv2.destroyWindow(MAIN_WINDOW_NAME)
    cv2.destroyWindow(DASHBOARD_WINDOW_NAME)

    # --- 8. NEW: Show & Save Session Summary Report ---
    print("Generating session summary...")

    # --- MODIFIED --- Increased canvas height to 720px for new info
    summary_canvas = np.zeros((720, 500, 3), dtype="uint8")
    summary_canvas.fill(255)  # White background

    # --- NEW: Calculate Total Durations ---
    total_attentive_time = 0.0
    total_non_attentive_time = 0.0
    if len(full_session_history) > 1:
        for i in range(1, len(full_session_history)):
            prev_time, _, prev_attentive, _ = full_session_history[i - 1]
            curr_time, _, _, _ = full_session_history[i]

            duration_of_segment = curr_time - prev_time

            # The state is determined by the *start* of the segment
            if prev_attentive:
                total_attentive_time += duration_of_segment
            else:
                total_non_attentive_time += duration_of_segment

    # --- NEW: Get Session Tag ---
    session_tag = get_session_tag(total_attentive_time, total_non_attentive_time, non_attentive_alert_count,
                                  emotions_map)

    # --- Title ---
    cv2.putText(summary_canvas, "Session Summary", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # --- NEW: Session Tag ---
    cv2.putText(summary_canvas, f"Session Tag: {session_tag}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # --- Alert Count ---
    cv2.putText(summary_canvas, f"Total Non-Attentive Alerts: {non_attentive_alert_count}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # --- NEW: Total Durations ---
    cv2.putText(summary_canvas, f"Total Attentive Time: {total_attentive_time:.1f}s", (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(summary_canvas, f"Total Non-Attentive Time: {total_non_attentive_time:.1f}s", (10, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # --- Emotion Bar Graph ---
    cv2.putText(summary_canvas, "Emotion Analysis:", (10, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    y_pos = 240
    total_emotions = sum(emotions_map.values())

    if total_emotions > 0:
        for emotion, count in emotions_map.items():
            percentage = count / total_emotions
            text = f"{emotion.capitalize()}:"
            cv2.putText(summary_canvas, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            bar_width_pixels = int(percentage * 350)
            cv2.rectangle(summary_canvas, (100, y_pos - 15), (100 + bar_width_pixels, y_pos + 5),
                          (255, 150, 50), -1)
            cv2.putText(summary_canvas, f"{(percentage * 100):.1f}%", (105 + bar_width_pixels, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            y_pos += 30
    else:
        cv2.putText(summary_canvas, "No emotion data was collected.", (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        y_pos += 30

    # --- MODIFIED: Full Session Timeline Graph (with new logic) ---
    graph_area_y_start = y_pos + 20  # Start graph below the bar chart
    graph_area_height = 150
    graph_area_y_end = graph_area_y_start + graph_area_height
    graph_area_x_start = 10
    graph_area_x_end = 490
    graph_area_width = graph_area_x_end - graph_area_x_start

    cv2.putText(summary_canvas, "Full Session Timeline:", (10, graph_area_y_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.rectangle(summary_canvas, (graph_area_x_start, graph_area_y_start), (graph_area_x_end, graph_area_y_end),
                  (0, 0, 0), 1)

    if len(full_session_history) > 1:
        session_start_time = full_session_history[0][0]
        session_end_time = full_session_history[-1][0]
        session_duration = session_end_time - session_start_time
        if session_duration == 0: session_duration = 1.0  # Avoid divide by zero

        # Draw the graph line by line
        for i in range(1, len(full_session_history)):
            # Get data for previous and current point
            prev_time, prev_emotion, _, _ = full_session_history[i - 1]
            curr_time, curr_emotion, curr_attentive, curr_duration = full_session_history[i]

            # Don't draw if there was a gap (no face)
            if prev_emotion == -1 or curr_emotion == -1:
                continue

            # --- Scale Y to fit the graph area ---
            y1_px = int(graph_area_y_end - (prev_emotion / 6.0) * graph_area_height)
            y2_px = int(graph_area_y_end - (curr_emotion / 6.0) * graph_area_height)

            # --- MODIFIED: Scale X based on TIME, not index ---
            x1_px = int(graph_area_x_start + ((prev_time - session_start_time) / session_duration) * graph_area_width)
            x2_px = int(graph_area_x_start + ((curr_time - session_start_time) / session_duration) * graph_area_width)

            # --- MODIFIED: Color (Attentiveness) with Alert state ---
            if curr_attentive:
                color = (0, 200, 0)  # Green
            else:
                if curr_duration > NON_ATTENTIVE_TIME_LIMIT:
                    color = (0, 0, 0)  # Black
                else:
                    color = (0, 0, 255)  # Red

            cv2.line(summary_canvas, (x1_px, y1_px), (x2_px, y2_px), color, 1)
    else:
        cv2.putText(summary_canvas, "Not enough data for timeline.", (10, graph_area_y_start + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # --- NEW: Legend for new graph ---
    cv2.rectangle(summary_canvas, (15, graph_area_y_end + 10), (25, graph_area_y_end + 20), (0, 200, 0), -1)
    cv2.putText(summary_canvas, "Attentive", (30, graph_area_y_end + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.rectangle(summary_canvas, (115, graph_area_y_end + 10), (125, graph_area_y_end + 20), (0, 0, 255), -1)
    cv2.putText(summary_canvas, "Not-Attentive", (130, graph_area_y_end + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                1)
    cv2.rectangle(summary_canvas, (235, graph_area_y_end + 10), (245, graph_area_y_end + 20), (0, 0, 0), -1)
    cv2.putText(summary_canvas, "Alerted", (250, graph_area_y_end + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # --- Final Instructions ---
    cv2.putText(summary_canvas, "Press any key to exit.", (10, summary_canvas.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # --- Save the summary canvas to a file ---
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        summary_filename = os.path.join(SNAPSHOT_DIR, f"summary_{timestamp}.png")
        cv2.imwrite(summary_filename, summary_canvas)
        print(f"Successfully saved summary report to: {summary_filename}")
    except Exception as e:
        print(f"Error: Failed to save summary report. {e}")

    # --- Show the summary window ---
    cv2.imshow("Session Summary", summary_canvas)
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()  # Final cleanup