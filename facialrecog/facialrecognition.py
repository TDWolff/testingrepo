import cv2
import os
import numpy as np

# Step 1: Prepare training data
data_folder_path = '/Users/torinwolff/Documents/GitHub/testingrepo/facialrecog/training'
face_images = [os.path.join(data_folder_path, f) for f in os.listdir(data_folder_path) if os.path.isfile(os.path.join(data_folder_path, f))]
faces = []
labels = []

for image_path in face_images:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    faces.append(img)
    labels.append(0)  # 0 for your face

# Step 2: Train the model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

# Step 3: Use the model to recognize faces
cap = cv2.VideoCapture(0)  # 0 for default camera

while True:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(roi_gray)

        if label == 0:  # 0 for your face
            print("Hello, world!")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('img', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()