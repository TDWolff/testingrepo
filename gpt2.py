import requests

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Get user input from the terminal
user_input = input("Enter your question or input text: ")

# Query the model with user input
output = query({
    "inputs": user_input,
})

# Display the entire model response
print("Model Response:")
print(output)
