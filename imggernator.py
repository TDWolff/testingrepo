import requests
from PIL import Image
import io
import os

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

# Get user input as a prompt
user_prompt = input("Enter prompt: ")

# Replace spaces with underscores in the user's input
filename = user_prompt.replace(" ", "_")

# Query the model with the user's prompt
image_bytes = query({"inputs": user_prompt})

# Convert the image bytes to a PIL Image
image = Image.open(io.BytesIO(image_bytes))

#Check if the current filename already exists somewhere in the folder
i = 1
while os.path.exists(f"{filename}.png"):
    filename = f"{filename} ({i})"
    i += 1

# Save the image with a filename based on the user's input (spaces replaced with underscores)
image.save(f"{filename}.png")
