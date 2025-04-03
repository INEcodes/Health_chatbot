import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import tensorflow as tf
import json
import streamlit as st
from nltk.tokenize import sent_tokenize
import tflearn
from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
groq_api = os.getenv('groq_api')
if not groq_api:
    st.error("Groq API key is missing in the environment variables.")
    exit()

client = Groq(api_key=groq_api)

# Load intents
with open("intents.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

# Data Preprocessing
words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# Stem and sort words
words = [stemmer.stem(w.lower()) for w in words if w not in ["?", "!", ".", ","]]
words = sorted(list(set(words)))
labels = sorted(labels)

# Training Data Preparation
def prepare_training_data():
    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = [1 if stemmer.stem(w.lower()) in [stemmer.stem(wrd) for wrd in doc] else 0 for w in words]
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    return np.array(training), np.array(output)

# Neural Network Model Creation
def create_model(training_data, output_data):
    tf.compat.v1.reset_default_graph()
    net = tflearn.input_data(shape=[None, len(training_data[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output_data[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)
    model.fit(training_data, output_data, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
    return model

# Bag of Words Function
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

# Function to analyze health using Groq API
def analyze_health_from_text(text, prompt):
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": f"Portfolio: {text}. You are a health chatbot. Provide medically accurate data, procedures, and the best possible quick therapy or remedy. Reference this as well: {prompt}"
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error during API request: {str(e)}"

# Streamlit Chat Function
def chat(model):
    st.title("Chat with the Health Bot!")
    st.write("Start talking with the bot! Type 'quit' to stop.")

    user_input = st.text_input("You: ")

    if user_input.lower() == "quit":
        st.write("Ending the chat...")
        return

    if user_input:
        # Predict intent
        results = model.predict([bag_of_words(user_input, words)])
        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
                response = responses[0]

                # Analyze using Groq API
                groq_response = analyze_health_from_text(user_input, response)

                st.markdown(f'<p style="color:rgb(34, 139, 34)">{groq_response}</p>', unsafe_allow_html=True)

# Main Function
def main():
    if os.path.exists("model.tflearn"):
        model = tflearn.DNN(tflearn.input_data(shape=[None, len(words)]))
        model.load("model.tflearn")
    else:
        training_data, output_data = prepare_training_data()
        model = create_model(training_data, output_data)

    chat(model)

# Run the Streamlit App
if __name__ == "__main__":
    main()
