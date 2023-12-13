from deepface import DeepFace
import os
import cv2
import numpy as np
import time

user_input = input("Do you want to register a new face (New/N) or recognize a face (Rec/R)? ")

if user_input.lower() == "new" or user_input.lower() == "n":
    print("Registering a new face...")
    # Capture a new image from the webcam
    cap = cv2.VideoCapture(0)
    user_name = input("Enter your name: ")
    new_folder_path = f"/Users/torinwolff/Documents/GitHub/testingrepo/facialrecog/training/{user_name}"
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    i = 1
    print("Look at the camera...")
    while i <= 20:  # capture 20 images
        ret, frame = cap.read()
        cv2.imshow('Capturing Image', frame)
        if i % 2 == 0:  # capture an image every other iteration
            cv2.imwrite(f"{new_folder_path}/{user_name} ({i//2}).jpg", frame)
        if cv2.waitKey(500) & 0xFF == ord('q'):  # wait 500 ms between images
            break
        i += 1
    cap.release()
    cv2.destroyAllWindows()
else:
    # Load the images from the training folder
    data_folder_path = '/Users/torinwolff/Documents/GitHub/testingrepo/facialrecog/training/TorinWolff'
    face_images = [os.path.join(data_folder_path, f) for f in os.listdir(data_folder_path) if os.path.isfile(os.path.join(data_folder_path, f))]

    # Create a dictionary to store the identities
    identities = {}

    for image_path in face_images:
        identity = os.path.splitext(os.path.basename(image_path))[0]
        identities[identity] = image_path

    # Capture a new image from the webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    # Increase the brightness
    frame = cv2.convertScaleAbs(frame, alpha=1, beta=50)

    cv2.imshow('Captured Image', frame)
    cv2.waitKey(1000)  # waits for 1000 ms
    cv2.destroyAllWindows()

    new_image_path = "/Users/torinwolff/Documents/GitHub/testingrepo/facialrecog/faces/new_image.jpg"
    cv2.imwrite(new_image_path, frame)
    cap.release()

    # Initialize a counter
    true_counter = 0

    for identity, db_img_path in identities.items():
        print(f"Verifying {identity}...")
        result = DeepFace.verify(new_image_path, db_img_path, model_name = "Facenet", enforce_detection = False)
        print(f"Is {identity} verified: ", result["verified"])

        # Increment the counter if the verification result is True
        if result["verified"]:
            true_counter += 1

    # Check if the counter is 2 or more
    if true_counter >= 2:
        print("It's the right person!")
    else:
        print("It's not the right person.")