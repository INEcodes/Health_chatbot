import nltk
import numpy as np
import tensorflow as tf
import random
import json
import pickle
import os
import PyPDF2  # Assuming you need this for PDF extraction
from dotenv import load_dotenv
import tflearn
from groq import Groq  # Assuming this is how to import the Groq client

# Load environment variables for Groq API
load_dotenv()

# Initialize the stemmer and other necessary components
stemmer = nltk.LancasterStemmer()

# Load intents data from JSON file
with open("intents.json", 'r', encoding='cp850') as f:
    data = json.load(f)

# Initialize lists to hold words, labels, and documents
words = []
labels = []
docs_x = []
docs_y = []

# Tokenize and process the intents data
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# Stem and sort words
words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
words = sorted(list(set(words)))

# Sort the labels
labels = sorted(labels)

# Prepare training and output data
training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# Define the neural network structure using tflearn
tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# Load the trained model (assuming it's already trained and saved as "model.tflearn")
model.load("model.tflearn")

# Function to convert a sentence into a bag of words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

# Initialize the Groq API client using the environment variable for the API key
groq_api = os.getenv('groq_api')
client = Groq(api_key=groq_api)

# Function to extract the text from the uploaded PDF
def extract_text_from_pdf(uploaded_file):
    # Reading the uploaded file directly from memory
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

# Function to analyze portfolio using Groq API
def analyze_health_from_text(text):
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": f"Portfolio: {text}. You are an health chatbot, u need to provide medically accurate data and procedures, Also provide the best possible quick therapy or remedy."
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )

        # Accessing the content directly from the 'message' object
        choices = completion.choices[0].message.content
        return choices
    
    except Exception as e:
        return f"Error during API request: {str(e)}"

# Main function to use the Groq API and output its result
def groq_output():
    print("Start talking with the bot! Type 'quit' when you want to quit")

    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        # Get the response from Groq API using the portfolio analysis function
        groq_response = analyze_health_from_text(inp)

        # Output the Groq API response
        print(f"Groq API Response: \n{groq_response}")

# Start the process
groq_output()