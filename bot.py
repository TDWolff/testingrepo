import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load intents from intents.json
import json

with open('intents.json', 'r') as file:
    intents = json.load(file)

# Extract patterns, responses, and contexts
patterns = []
responses = []
contexts = []
for intent in intents['intents']:
    if 'patterns' in intent and 'responses' in intent:
        patterns.extend(intent['patterns'])
        responses.extend(intent['responses'])
        contexts.extend(intent['context'] * len(intent['patterns']))  # Repeat the context for each pattern

# Ensure each pattern has a corresponding response
if len(patterns) != len(responses):
    raise ValueError("Mismatch in the number of patterns and responses.")

# Tokenize patterns
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(patterns)
total_words = len(tokenizer.word_index) + 1

# Create input sequences and labels
input_sequences = tokenizer.texts_to_sequences(patterns)
input_sequences_padded = pad_sequences(input_sequences, padding='post')
labels = np.array(tf.keras.utils.to_categorical(np.arange(len(responses)), num_classes=total_words))

# Check consistency of data
if len(input_sequences_padded) != len(labels):
    raise ValueError("Mismatch in the number of input sequences and labels.")

# Build the model
model = Sequential([
    Embedding(total_words, 100, input_length=input_sequences_padded.shape[1]),
    LSTM(100),
    Dense(128, activation='relu'),
    Dense(total_words, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(input_sequences_padded, labels, epochs=50, verbose=1)

# Function to generate a response from the model
def generate_response(input_text, current_context):
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_sequence_padded = pad_sequences(input_sequence, maxlen=input_sequences_padded.shape[1], padding='post')
    predicted_probabilities = model.predict(input_sequence_padded)[0]
    predicted_word_index = np.argmax(predicted_probabilities)
    predicted_word = tokenizer.index_word[predicted_word_index]

    # Check if there is a context switch
    if current_context:
        context_index = contexts.index(current_context)
        response_context = intents['intents'][context_index // len(intent['patterns'])]['context'][0]
        return predicted_word, response_context
    else:
        return predicted_word, None

# Test the chatbot
current_context = None
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    response, current_context = generate_response(user_input, current_context)
    print(f"ChatBot: {response}")

    # If there's a context switch, update the current context
    if current_context:
        print(f"Context Switch: {current_context}")
