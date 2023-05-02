from __future__ import print_function
from base64 import b16decode
from dataclasses import replace
from lib2to3.pgen2.tokenize import tokenize
from pickletools import read_string1
from tabnanny import verbose
from telnetlib import PRAGMA_HEARTBEAT
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import re
import matplotlib.pyplot as plt
import fasttext
from text2vec import ftxtGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from  attentionLayer import  TransformerBlock
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Input, Dropout, Conv1D
from keras.layers import LSTM, Bidirectional
from tensorflow.keras import Model
import tensorflow as tf
from sklearn.model_selection import train_test_split



# import definitions
modelpath = '/home/hanieh/Desktop/NLP/result/fil9.bin'
vecsize = 128



test = ftxtGenerator(modelpath, vecsize)

res = pd.read_csv('/home/hanieh/Desktop/NLP/Project/dataset.csv')
winsize = 15
labels = []
x_train = []
for i in range(res.shape[0]):
    text = res['tweet'].iloc[i]
    batch = test.text2batch(text=text, winSize=winsize)
    # print(len(batch))
    if len(batch) == 0:
        print(i)
        continue
    # lbl = [res['news_category'].iloc[i]] * batch.shape[0]
    lbl = res['label'].iloc[i]
    labels.append(lbl)
    x_train.append(batch)
# print(labels)
x_train = np.array(x_train)
le = LabelEncoder()

label = le.fit_transform(labels)

y_train = to_categorical(label,num_classes=3)
#//////////////////////////////////////////////////////////////////

max_features = 20000
maxlen = winsize  # cut texts after this number of words (among top max_features most common words)
batch_size = 64


x_train_t = sequence.pad_sequences(x_train, maxlen=maxlen)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, shuffle=True)

inp = Input(shape=(winsize, 128))
b1 = Bidirectional(LSTM(256, return_sequences=True,recurrent_dropout=0.2), merge_mode='concat')(inp)
# con1 = Conv1D(512, 3)(b1)
tb ,w1 = TransformerBlock(512, 8, 250)(b1)
lst = Bidirectional(LSTM(128, recurrent_dropout=0.2, return_sequences=False))(tb)
dns1 = Dense(256, activation='tanh')(lst)
dp1 = Dropout(0.3)(dns1)
dns2 = Dense(128, activation='tanh')(dp1)
dp2 = Dropout(0.3)(dns2)
outp = Dense(3, activation='softmax')(dp2)

model = Model(inputs=[inp], outputs=[outp])

# try using different optimizers and different optimizer configs
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

checkpoint_filepath = '/home/hanieh/Desktop/NLP/result/BLSTM_CNN_before.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1,
    save_best_only=True)

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15, validation_data=(x_val, y_val), callbacks=[model_checkpoint_callback])
model.load_weights(checkpoint_filepath)
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
y_pred = model.predict(x_test, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

y_pred_bool = to_categorical(y_pred_bool,num_classes=3)
print(y_pred_bool.shape)
print(y_test.shape)
print(classification_report(y_test, y_pred_bool))
print(precision_score(y_test, y_pred_bool , average="macro"))
print(recall_score(y_test, y_pred_bool , average="macro"))
print(f1_score(y_test, y_pred_bool , average="macro"))