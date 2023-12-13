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
    # Capture a new image from the webcam
    cap = cv2.VideoCapture(0)
    print("Look at the camera...")
    ret, frame = cap.read()
    new_image_path = "/Users/torinwolff/Documents/GitHub/testingrepo/facialrecog/new_image.jpg"
    cv2.imwrite(new_image_path, frame)
    cap.release()

    # Load the images from the training folder and its subfolders
    data_folder_path = '/Users/torinwolff/Documents/GitHub/testingrepo/facialrecog/training'
    for root, dirs, files in os.walk(data_folder_path):
        face_images = [os.path.join(root, file) for file in files if file.endswith((".jpg", ".png"))]  # add more file types if needed
        if not face_images:  # skip if no images in the folder
            continue

        # Initialize a counter
        true_counter = 0

        for image_path in face_images:
            print(f"Verifying {image_path}...")
            result = DeepFace.verify(new_image_path, image_path, model_name = "Facenet", enforce_detection = False)
            print(f"Is {image_path} verified: ", result["verified"])

            # Increment the counter if the verification result is True
            if result["verified"]:
                true_counter += 1

        # Check if the counter is 2 or more
        if true_counter >= 2:
            print(f"It's the right person, Welcome {os.path.basename(root)}!")
            break
    else:
        print("It's not the right person.")