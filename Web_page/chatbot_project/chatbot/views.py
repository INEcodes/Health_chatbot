from django.shortcuts import render

# Create your views here.
# chatbot/views.py
import json
import nltk
import numpy as np
import tensorflow as tf
import tflearn
from nltk.stem.lancaster import LancasterStemmer
from django.shortcuts import render
from django.http import JsonResponse

# Initialize the stemmer
stemmer = LancasterStemmer()

# Load intents file
with open('chatbot/intents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Initialize lists for the words, labels, and the training data
words = []
labels = []
docs_x = []
docs_y = []

# Prepare the dataset for training
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

# Prepare the training data
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

# Build the neural network
tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load('chatbot/model.tflearn')

# Function to process the input and return a prediction
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

# View function to render the chatbot interface
def index(request):
    return render(request, 'chatbot/index.html')

# API endpoint for chatbot response
def chat(request):
    if request.is_ajax():
        user_message = request.GET.get('message')
        if user_message:
            # Predict the response using the trained model
            results = model.predict([bag_of_words(user_message, words)])
            results_index = np.argmax(results)
            tag = labels[results_index]

            # Find the response for the predicted tag
            response = ""
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    response = tg['responses'][0]
                    break
            return JsonResponse({'response': response})

    return JsonResponse({'error': 'No message received'})

