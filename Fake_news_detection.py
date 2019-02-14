import numpy as np
import pandas as pd
import json as j
import urllib
import gzip
import nltk

nltk.download('stopwords')
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
#!pip install gensim
from gensim.models import KeyedVectors
from nltk.corpus import stopwords

from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import re
# import sys

# reload(sys)
# sys.setdefaultencoding('utf8')


import string
from sklearn.feature_extraction.text import CountVectorizer


def clean_text(text):
    # text = [w.strip() for w in text.readlines()]
    # text.decode('unicode_escape').encode('ascii','ignore')
    text = str(text)
    # text = text.decode("utf8")

    text = text.split()
    words = []
    for word in text:
        exclude = set(string.punctuation)
        word = ''.join(ch for ch in word if ch not in exclude)
        if word in stops:
            continue
        try:
            words.append(ps.stem(word))
        except UnicodeDecodeError:
            words.append(word)
    text = " ".join(words)

    return text.lower()


# Process data

stops = set(stopwords.words("english"))

ps = PorterStemmer()

dataset = 'https://www.dropbox.com/s/bxix43wtl06vs74/fake_or_real_news.csv?dl=1'

f = pd.read_csv(dataset)

f.label = f.label.map(dict(REAL=1, FAKE=0))
#print(f)



# ---------- text cleaning ---------
f = f[1:100]

X_train, X_test, y_train, y_test = train_test_split(f['title'], f.label, test_size=0.2)

X_cleaned_train = [clean_text(x) for x in X_train]

X_cleaned_test = [clean_text(x) for x in X_test]



print(X_cleaned_train[0])



#Tokenizer

#Here I tokenize the data to assign indices to words, and filter out infrequent words. This allows me to generate sequences for my training and testing data.

import tokenize
from keras.preprocessing.text import Tokenizer

MAX_NB_WORDS = 20000
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(X_cleaned_train + X_cleaned_test)
print('Finished Building Tokenizer')

train_sequence = tokenizer.texts_to_sequences(X_cleaned_train)
print('Finished Tokenizing Training')

test_sequence = tokenizer.texts_to_sequences(X_cleaned_test)
print('Finished Tokenizing Training')


#Embedding Matrix
#I am using an embedding matrix to extract the semantic information from the words in each title.


from gensim.models import KeyedVectors
EMBEDDING_FILE = 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

nb_words = min(20000, len(word_index))

embedding_matrix = np.zeros((nb_words, 300))


for word, i in word_index.items():
    try:
      embedding_vector = word2vec.word_vec(word)
      if embedding_vector is not None and i < 7000:
        embedding_matrix[i] = embedding_vector
    except (KeyError, IndexError) as e:
      continue


#Building the Model
#Here I create the model using an Embedding layer, LSTM, Dropout, and Dense layers. I am going to run my data on 20 epochs.


from keras.models import Sequential

from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences

kVECTORLEN = 50

model = Sequential()
model.add(Embedding(5000, 500, input_length=50))
model.add(LSTM(125))
model.add(Dropout(0.4))
model.add(Dense(1, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())


#test_sequence

train_sequence = sequence.pad_sequences(train_sequence, maxlen=50)
test_sequence = sequence.pad_sequences(test_sequence, maxlen=50)

history = model.fit(train_sequence, y_train, validation_data=(test_sequence, y_test), epochs=20, batch_size=64)

scores = model.evaluate(test_sequence, y_test, verbose=0)
accuracy = (scores[1]*100)

print("Accuracy: {:.2f}%".format(scores[1]*100))



plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


