import random
import json
import pickle
from unittest import result
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer #lemmatizer helps to recognize different word as one
from tensorflow.keras.models import load_model

lem = WordNetLemmatizer()
intents = json.loads(open('C:\\Users\\User\\Desktop\\Natural language processed AI chat bot\\intents.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lem.lemmatize(word)for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] == 1
    return np.array(bag)

def predict_class(sentence):
    bag = bag_of_words(sentence)
    res = model.predict(np.array([bag]))[0]
    Error_treshold = 0.25
    result = [[i,r] for i,r in enumerate(res) if r > Error_treshold]

    result.sort(key=lambda x:x[1], reverse=True)
    return_li = []
    for r in result:
        return_li.append({'intent':classes[r[0]], 'probability': str(r[1])})
    return return_li

def get_response(intents_li,intents_json):
    tag = intents_li[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("CHAT WITH ME !")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)




