#make sure to use pip install tensorflow==2.7.0
import random
import json
import pickle
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer #lemmatizer helps to recognize different word as one

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD


lem = WordNetLemmatizer()
intents = json.loads(open('C:\\Users\\User\\Desktop\\Natural language processed AI chat bot\\intents.json').read())

words = []
classes = []
docs = []
ignore = ['?','*','!','.',',',':',')','(']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_li = nltk.word_tokenize(pattern) # separatly recognize every word in a line
        words.extend(word_li) # extend takes content and append to the list
        docs.append((word_li,intent['tag']))# append takes list and append to the list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
# print(docs) - this will detect the main catagory of words ex: (['what', 'can', 'you', 'give', 'me', '?'], 'shop')

words = [lem.lemmatize(word) for word in words if word not in ignore]
words = sorted(set(words)) #set eleminate the duplicates

classes = sorted(set(classes))
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# as neural networks do not understand words so, using numbers

training = []
output_empty = [0]*len(classes)

#loop to insert into the training list
for doc in docs :
   bag = []
   word_patterns = doc[0]
   word_patterns = [lem.lemmatize(word.lower()) for word in word_patterns]
   for word in words:
       bag.append(1) if word in word_patterns else bag.append(0)

   output_row = list(output_empty)
   output_row[classes.index(doc[1])] = 1
   training.append([bag,output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])

#neural network model

model = Sequential()
#layers
model.add(Dense(128,input_shape=(len(train_x[0]),), activation='relu' ))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200,batch_size=5,verbose=1)
model.save('chatbot_model.h5',hist)
print('DONE')

