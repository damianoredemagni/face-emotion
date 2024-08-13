import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import cv2
import numpy as np
from deepface import DeepFace
from collections import defaultdict
import datetime

# Define a variable called 'deepface' and assign it to an instance of the DeepFace class
deepface = DeepFace.build_model("Emotion")

# Load the video file
cap = cv2.VideoCapture('test_face.mp4')
# Define the face cascade classifier

# Create a dictionary to keep track of emotion counts
emotion_counts = defaultdict(int)
# Create a dictionary to keep track of emotional anomalies
emotion_timestamps = defaultdict(list)

# Threshold for emotion detection to mark anomalies
emotion_threshold = 5  # This is a hypothetical threshold for detection spikes
# Total number of detected faces
total_faces = 0

# And replace the face detection loop with:
faces = DeepFace.extract_faces(frame, detector_backend='opencv')
for face in faces:
    if face['confidence'] > 0.9:  # You can adjust this threshold
        face_roi = face['face']
        try:
            emotions = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

    # Get the current timestamp in HH:MM:SS format
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        total_faces += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Crop the face from the frame for emotion detection
        face_roi = frame[y:y + h, x:x + w]

        # Detect emotions in the cropped face using DeepFace library
        try:
            emotions = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            # Print the detected emotions
            if emotions:
                for emotion in emotions:
                    emotion_name = emotion['dominant_emotion']
                    emotion_counts[emotion_name] += 1

                    # Check for spikes in emotion counts
                    if emotion_counts[emotion_name] > emotion_threshold:
                        if current_time not in [ts["time"] for ts in emotion_timestamps[emotion_name]]:
                            emotion_timestamps[emotion_name].append({"time": current_time, "count": emotion_counts[emotion_name]})

                    # You can add the emotion text to the rectangle as well
                    cv2.putText(frame, f"{emotion_name} ({emotion_counts[emotion_name]})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        except Exception as e:
            print("Error in emotion detection:", e)

    # Display the frame
    cv2.imshow('Video', frame)

    # Check for user input to stop the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate percentages of each emotion
if total_faces > 0:
    print("Emotional Breakdown:")
    for emotion, count in emotion_counts.items():
        percentage = (count / total_faces) * 100
        print(f"{emotion}: {percentage:.2f}%")
else:
    print("No faces detected.")

# Print emotional anomalies
print("\nEmotional Anomalies (Timestamps):")
for emotion, timestamps in emotion_timestamps.items():
    for entry in timestamps:
        print(f"{emotion} anomaly detected at {entry['time']} (Count: {entry['count']})")

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
