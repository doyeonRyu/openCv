# PYTHON- OpenCV Face Detection while capturing video from webcam

    # This code is for detecting faces in a video stream from a webcam using OpenCV.
    # The code uses the Haar Cascade Classifier to detect faces in the video stream.
    # 웹캠으로 얼굴을 감지해 사각형을 그리고 얼굴 개수를 화면에 출력함
    # 웹캠을 비디오로 저장
    # 

import cv2
import sys

# Read from the camera
cap = cv2.VideoCapture(0)
# 0: This is for the first webcam. Try other numbers if you have multiple webcams.

# Set the size of the frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.05, 4, minSize=(200, 200))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Display the number of faces detected
        face_count = len(faces)
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
