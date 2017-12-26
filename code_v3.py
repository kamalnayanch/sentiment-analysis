import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding, Input, Lambda
from keras.layers.wrappers import Bidirectional
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD, Adagrad
from keras.layers.convolutional import Conv1D,Conv2D
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D, AveragePooling1D,MaxPooling2D, GlobalMaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM, GRU
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
import numpy as np
import tensorflow as tf
import re
import keras.callbacks
import sys
import os
import h5py


from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures, scale
from sklearn.ensemble import AdaBoostRegressor


def binarize(x):
    sz = len(chars)
    #print(tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1)))
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))


def binarize_outshape(in_shape):
    #print (in_shape[0], in_shape[1], len(chars),1)
    return (in_shape[0], in_shape[1], len(chars),1)

# record history of training
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))


total = len(sys.argv)
cmdargs = str(sys.argv)

print ("Script name: %s" % str(sys.argv[0]))
checkpoint = None
if len(sys.argv) == 2:
    if os.path.exists(str(sys.argv[1])):
        print ("Checkpoint : %s" % str(sys.argv[1]))
        checkpoint = str(sys.argv[1])

training_data = pd.read_csv("preprocessedanger-ratings-0to1.train.txt",delimiter="\t", names=["ID","Text","Emotion","Intensity"])
validation_data = pd.read_csv("preprocessedanger-ratings-0to1.dev.gold.txt", delimiter="\t", names=["ID","Text","Emotion","Intensity"])
testing_data = pd.read_csv("preprocessedanger-ratings-0to1.test.gold.txt", delimiter="\t", names=["ID","Text","Emotion","Intensity"])

docs = []
sentiments = []

for cont, sentiment in zip(training_data.Text, training_data.Intensity):
    docs.append(cont)
    sentiments.append(float(sentiment))

docs_val = []
sentiments_val = []
for cont, sentiment in zip(validation_data.Text, validation_data.Intensity):
    docs_val.append(cont)
    sentiments_val.append(float(sentiment))

docs_test = []
sentiments_test = []
for cont, sentiment in zip(testing_data.Text, testing_data.Intensity):
    docs_test.append(cont)
    sentiments_test.append(float(sentiment))

num_sent = []
txt = ''
for doc in docs:
    num_sent.append(len(doc))
    for s in doc:
        txt += s

chars = set(txt)

print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print('Sample doc{}'.format(docs[400]))

maxlen = 150

X = np.ones((len(docs), maxlen), dtype=np.int64) * -1
y = np.array(sentiments)

X_val = np.ones((len(docs_val),maxlen),dtype = np.int64)*-1
y_val = np.array(sentiments_val)

X_test = np.ones((len(docs_test),maxlen),dtype = np.int64)*-1
y_test = np.array(sentiments_test)

for i, doc in enumerate(docs):
  for t,char in enumerate(doc[-maxlen:]):
    X[i,(maxlen - 1 - t)] = char_indices[char]

for i, doc in enumerate(docs_val):
  for t,char in enumerate(doc[-maxlen:]):
    if(char_indices.get(char)==None):
      print(char)
      X_val[i,(maxlen - 1 - t)] = len(chars)+1
    else:
      X_val[i,(maxlen - 1 - t)] = char_indices[char]
      #print (char_indices[char])

for i, doc in enumerate(docs_test):
  for t,char in enumerate(doc[-maxlen:]):
    if(char_indices.get(char)==None):
      print(char)
      X_test[i,(maxlen - 1 - t)] = len(chars)+1
    else:
      X_test[i,(maxlen - 1 - t)] = char_indices[char]
     # print (char_indices[char])




print('Sample chars in X:{}'.format(X[400]))
print('y:{}'.format(y[400]))

ids = np.arange(len(X))
np.random.shuffle(ids)

# shuffle
X = X[ids]
y = y[ids]


def get_model():
  filter_length = [5,5,3,3, 3]
  nb_filter = [256, 256, 128,128,64]
  pool_length = 2

  # sentence input
  in_sentence = Input(shape=(maxlen,), dtype='int64')
  # char indices to one hot matrix, 1D sequence to 2D 
  embedded = Lambda(binarize, output_shape=binarize_outshape)(in_sentence)
  # embedded: encodes sentence
  print ('hi')
  for i in range(len(nb_filter)):
      embedded = Conv2D(filters=nb_filter[i],
                        kernel_size=(filter_length[i],filter_length[i]),
                        padding='valid',
                        activation='relu',
                        kernel_initializer='glorot_normal',
                        strides=(1,1))(embedded)

      embedded = Dropout(0.1)(embedded)
      embedded = MaxPooling2D(pool_size=(pool_length,pool_length))(embedded)
  print ('hi')
  # bi_lstm_sent = \
  #     Bidirectional(LSTM(128, return_sequences=True, dropout=0.15, recurrent_dropout=0.15, implementation=0))(embedded)

  # bi_lstm_sent_1 = \
  #     Bidirectional(LSTM(64, return_sequences=True, dropout=0.15, recurrent_dropout=0.15, implementation=0))(bi_lstm_sent)

  # bi_lstm_sent_2 = \
  #     Bidirectional(LSTM(32, return_sequences=False, dropout=0.15, recurrent_dropout=0.15, implementation=0))(bi_lstm_sent_1)

  # sent_encode = merge([forward_sent, backward_sent], mode='concat', concat_axis=-1)
  #output = Dropout(0.3)(bi_lstm_sent_2)
  output = Dense(128, activation='relu')(embedded)
  output = Dropout(0.3)(output)
  output = Dense(1, activation='sigmoid')(output)
  # sentence encoder
  model = Model(inputs=in_sentence, outputs=output)
  print(model.summary())
  model.compile(loss='mean_squared_error', optimizer='adam')
  print(model.summary())

  return model


# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

estimator = KerasRegressor(build_fn = get_model,epochs =50,batch_size=32,verbose=1,callbacks=callbacks_list)
estimator.fit(X,y,validation_data = (X_val,y_val))
train_prediction = estimator.predict(X)
print(pearsonr(train_prediction,y))
print(spearmanr(train_prediction,y))

val_prediction = estimator.predict(X_val)
print(pearsonr(val_prediction,y_val))
print(spearmanr(val_prediction,y_val))


test_prediction = estimator.predict(X_test)
print(pearsonr(test_prediction,y_test))
print(spearmanr(test_prediction,y_test))
