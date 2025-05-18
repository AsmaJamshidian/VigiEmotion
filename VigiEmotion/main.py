# Import necessary libraries
import os
import cv2
import csv
import requests
from datetime import datetime
from deepface import DeepFace

# Telegram bot token and chat ID configuration
TELEGRAM_TOKEN = "your_telegram_bot_token"
CHAT_ID = "your_chat_id"

# Function to send an image to Telegram when a suspicious person is detected
def send_photo_to_telegram(image_path, caption="⚠️ Suspicious person detected"):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(image_path, 'rb') as photo:
        files = {'photo': photo}
        data = {'chat_id': CHAT_ID, 'caption': caption}
        requests.post(url, files=files, data=data)

# Define colors for each detected emotion to display on the video frame
emotion_colors = {
    "happy": (0, 255, 255),
    "sad": (255, 0, 0),
    "angry": (0, 0, 255),
    "surprise": (255, 255, 0),
    "fear": (128, 0, 128),
    "disgust": (0, 128, 0),
    "neutral": (200, 200, 200)
}

# Initialize a CSV log file to store facial analysis data
log_file = "log.csv"
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Name', 'Age', 'Gender', 'Emotion'])

# Start capturing video from the default webcam
cap = cv2.VideoCapture(0)

# Track already saved unknown faces to prevent duplicate alerts
saved_faces = set()

# Main loop to read and process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if the frame is not captured properly

    try:
        # Analyze the frame to detect emotions, age, and gender
        results = DeepFace.analyze(
            frame,
            actions=['emotion', 'age', 'gender'],
            enforce_detection=False
        )

        # Loop through each detected face
        for idx, face in enumerate(results):
            emotion = face['dominant_emotion']
            age = face['age']
            gender = face['gender']
            if isinstance(gender, dict):  # Handle gender prediction as a dictionary
                gender = max(gender, key=gender.get)

            # Get facial region coordinates
            region = face['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            face_crop = frame[y:y + h, x:x + w]
            color = emotion_colors.get(emotion, (0, 255, 0))
            approx_age = int(round(age))
            age_range = f"Age: {approx_age - 1}-{approx_age + 1}"

            # Try to find a match in the known faces database
            try:
                matches = DeepFace.find(face_crop, db_path="known/", enforce_detection=False)
                is_known = len(matches[0]) > 0
                person_name = os.path.basename(matches[0].iloc[0]['identity']).split('.')[0] if is_known else "Unknown"
            except:
                is_known = False
                person_name = "Unknown"

            # Draw rectangle and annotations on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(frame, f"Emotion: {emotion}", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Gender: {gender}", (x, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, age_range, (x, y + h + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Save and alert if the face is unknown and not previously saved
            if not is_known:
                face_key = f"{approx_age}_{gender}_{emotion}"
                if face_key not in saved_faces:
                    saved_faces.add(face_key)
                    if not os.path.exists("unknown_faces"):
                        os.makedirs("unknown_faces")
                    filename = f"unknown_faces/unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, frame)
                    send_photo_to_telegram(filename)

            # Append data to the log CSV file
            with open(log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    person_name, approx_age, gender, emotion
                ])

    except Exception as e:
        print("Error:", e)

    # Show the video frame with annotations
    cv2.imshow("DeepFace Real-time Emotion", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit loop on pressing 'q'

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
