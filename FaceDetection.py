# PYTHON- OpenCV Face Detection while capturing video from webcam

    # This code is for detecting faces in a video stream from a webcam using OpenCV.
    # The code uses the Haar Cascade Classifier to detect faces in the video stream.

    # Detect faces from webcam, draw a rectangle and print the number of faces on the screen.
    # Save the webcam as a video.
    # Save ['Frame', 'Timestamp', 'face_count'] data as a csv file on a frame-by-frame basis.

import cv2
import sys
from datetime import datetime
import pandas as pd

# Read from the camera
cap = cv2.VideoCapture(0)
# 0: This is for the first webcam. Try other numbers if you have multiple webcams.

# Set the size of the frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize a list to store detection data
detections = []

frame_number = 0

# Save the video as output.avi
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS) # 30.0
out = cv2.VideoWriter('output.avi', fourcc, fps, (640, 480))

# Load the face detector
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Check if the cascade file is loaded correctly
if face_cascade.empty():
    print(f"Error loading cascade file. Please ensure the file path is correct: {face_cascade_path}")
    sys.exit()

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame_number += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.05, 4, minSize=(200, 200))

        # Calculate the number of faces detected
        face_count = len(faces)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# if you want to save like this, you can add the following code
# df = pd.DataFrame(detections, columns=['Frame', 'Timestamp', 'Face_Count', 'X', 'Y', 'Width', 'Height'])

#        if face_count == 0:
#            # Append data with no face detected
#            detections.append([frame_number, current_time, face_count, None, None, None, None])
#        else:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Append detection data to the list with the current timestamp
            detections.append([frame_number, current_time, face_count])
    
        # Display the number of faces detected
        cv2.putText(frame, f'Faces: {face_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Write the frame into the file 'output.avi'
        out.write(frame)

        cv2.imshow('Video', frame)

        # Press 'q' on the keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if the job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

# Create a DataFrame from the detections list
df = pd.DataFrame(detections, columns=['Frame', 'Timestamp', 'Face_Count'])
# df = pd.DataFrame(detections, columns=['Frame', 'Timestamp', 'Face_Count', 'X', 'Y', 'Width', 'Height'])

# Save the DataFrame to a CSV file
df.to_csv('face_detections.csv', index=False)

print("Face detection data saved to 'face_detections.csv'")