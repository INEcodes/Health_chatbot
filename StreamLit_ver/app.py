import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import tensorflow as tf
import random
import json
import streamlit as st
from nltk.tokenize import sent_tokenize
import pickle
import json

with open("intents.json",'r',encoding='cp850') as f:
    data = json.load(f)

import tflearn
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
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

words = [stemmer.stem(w.lower()) for w in words if  w not in "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x,doc in enumerate(docs_x):
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

tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch = 1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

def chat():
    # Streamlit app layout and title
    st.title("Chat with the Bot!")
    st.write("Start talking with the bot! Type 'quit' to stop.")

    # Text input for user message
    user_input = st.text_input("You: ")

    if user_input.lower() == "quit":
        st.write("Ending the chat...")
        return

    if user_input:
        # Model prediction
        results = model.predict([bag_of_words(user_input, words)])
        results_index = np.argmax(results)
        tag = labels[results_index]

        # Find responses
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
                response_list = sent_tokenize(str(responses[0]))
                colors = [31, 32, 33, 34, 35]

                # Display the responses with color variation in Streamlit
                for i, response in enumerate(response_list):
                    color_index = i % len(colors)
                    color_code = colors[color_index]
                    st.markdown(f'<p style="color:rgb({color_code},{color_code},{color_code})">{response}</p>', unsafe_allow_html=True)

                
chat()