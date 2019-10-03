## Imports
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

## Data
sc = pd.read_csv('train-balanced-sarcasm.csv')


## Preprocessing
# Drop all empty comments
sc.dropna(subset=['comment'], inplace=True)

# TODO: PRUNE DATA
# TODO: Filter highest counted words (e.g. and, is, that) ~20%?
# TODO: Lowercase capitalized words but leave all caps words (i.e. Trump = trump, TRUMP != trump)


## Split
sc_train, sc_test, y_train, y_test = train_test_split(sc['comment'].values, sc['label'].values, test_size=0.3, random_state=42)


## KERAS

# Count Vectorizer
# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(min_df=0, lowercase=False)
# vectorizer.fit(sc_train)

# x_train = vectorizer.transform(sc_train)
# x_test = vectorizer.transform(sc_test)

# input_dim = x_train.shape[1]

n_words = 12000
maxlen = 100

# Tokenizer
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=n_words)
tokenizer.fit_on_texts(sc_train)

x_train = tokenizer.texts_to_sequences(sc_train)
x_test = tokenizer.texts_to_sequences(sc_test)

vocab_size = len(tokenizer.word_index) + 1

# Pad_sequences
from keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(x_train, maxlen=150)
x_test = pad_sequences(x_test, maxlen=150)
print(sc_train[3])
print(x_train[3])
print(vocab_size)

# Neural Network
from keras.models import Sequential
from keras import layers

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, output_dim=75, input_length=150))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, epochs=20, verbose=True, validation_data=(x_test, y_test), batch_size=50)

loss, accuracy = model.evaluate(x_train, y_train, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test, y_test, verbose=True)
print("Testing Accuracy:  {:.4f}".format(accuracy))


# num_words = 0
# pad_sequences, padding=post, maxlen=100
# output_dim=50,
# Layers.Flatten
# Training Accuracy: 0.8836
# Testing Accuracy: 0.6781


# num_words = 12000, maxlen = 150
# pad_sequences, prepend, maxlen=0
# output_dim=75,
# Epoch 8/20
# accuracy: 0.7792 - val_loss: 0.5815 - val_accuracy: 0.7069
# Training Accuracy: 0.8534
# Testing Accuracy:  0.6928
