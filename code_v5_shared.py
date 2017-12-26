# This model works on the concept of using shared layers

import numpy as np
import html
import re
from nltk import word_tokenize
from nltk import bigrams
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize.casual import TweetTokenizer


#import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding, Input
from keras.layers.wrappers import Bidirectional
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD, Adagrad
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM, GRU
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from keras.models import load_model

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
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM, GRU
from keras.preprocessing import sequence

import numpy as np
import tensorflow as tf
import re
import keras.callbacks
import sys
import os


from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures, scale
from sklearn.ensemble import AdaBoostRegressor


class Tweet(object):
	
	def __init__(self,id,text,emotion, intensity):
		self.id = id
		self.text = text
		self.emotion = emotion
		self.intensity = intensity

def read_training_data (training_filename):
	
	list_train = list();
	with open (training_filename) as file_input:
		for line in file_input:
			line = line.strip()
			array_splits = line.split('\t')
			list_train.append(Tweet(array_splits[0],preprocess_string(array_splits[1]),array_splits[2],float(array_splits[3])))

	print(len(list_train))
	return list_train;

def write_training_file(training_list,training_preprocessed):
	thefile = open(training_preprocessed, 'w')
	for item in training_list:
		thefile.write("%s\t%s\t%s\t%f\n" % (item.id,item.text,item.emotion,item.intensity))

	thefile.close()
	return
def preprocess_string(string):

	string = html.unescape(string)
	string = string.replace("\\n"," ")
	string = string.replace("_NEG","")
	string = string.replace("_NEGFIRST", "")
	string = re.sub(r"@[A-Za-z0-9_s(),!?\'\`]+", "", string) # removing any twitter handle mentions
	string = re.sub(r"#", "", string)
	string = re.sub(r"\*", "", string)
	string = re.sub(r"\'s", "", string)
	string = re.sub(r"\'m", " am", string)
	string = re.sub(r"\'ve", " have", string)
	string = re.sub(r"n\'t", " not", string)
	string = re.sub(r"\'re", " are", string)
	string = re.sub(r"\'d", " would", string)
	string = re.sub(r"\'ll", " will", string)
	string = re.sub(r",", "", string)
	string = re.sub(r"!", " !", string)
	string = re.sub(r"\(", "", string)
	string = re.sub(r"\)", "", string)
	string = re.sub(r"\?", " ?", string)
	string = re.sub(r'[^\x00-\x7F]',' ', string)
	string = re.sub(r"\s{2,}", " ", string)
	string = string.rstrip(',|.|;|:|\'|\"')
	string = string.lstrip('\'|\"')

	return string.lower() # remove_stopwords(string.strip().lower())
	
def remove_stopwords(string):
	temp = stopwords.words('english')
	split_string = \
	[word for word in string.split()
		if word not in temp]

	return " ".join(split_string)


def lstm_model(training_list_anger,validation_list_anger,test_list_anger,training_list_fear,validation_list_fear,test_list_fear,training_list_joy,validation_list_joy,test_list_joy,training_list_sadness,validation_list_sadness,test_list_sadness):

	anger_tweets_train = list()
	anger_score_train = list()
	total_dataset = list()
	for tweet in training_list_anger:
		anger_tweets_train.append(tweet.text);
		anger_score_train.append(float(tweet.intensity));
		total_dataset.append(tweet.text)

	anger_tweets_val = list();
	anger_score_val = list();
	for tweet in validation_list_anger:
		anger_tweets_val.append(tweet.text);
		anger_score_val.append(float(tweet.intensity));
		total_dataset.append(tweet.text)

	anger_tweets_test = list();
	anger_score_test = list();
	for tweet in test_list_anger:
		anger_tweets_test.append(tweet.text);
		anger_score_test.append(float(tweet.intensity));
		total_dataset.append(tweet.text)


	fear_tweets_train = list()
	fear_score_train = list()
	for tweet in training_list_fear:
		fear_tweets_train.append(tweet.text);
		fear_score_train.append(float(tweet.intensity));
		total_dataset.append(tweet.text)

	fear_tweets_val = list();
	fear_score_val = list();
	for tweet in validation_list_fear:
		fear_tweets_val.append(tweet.text);
		fear_score_val.append(float(tweet.intensity));
		total_dataset.append(tweet.text)

	fear_tweets_test = list();
	fear_score_test = list();
	for tweet in test_list_fear:
		fear_tweets_test.append(tweet.text);
		fear_score_test.append(float(tweet.intensity));
		total_dataset.append(tweet.text)

	joy_tweets_train = list()
	joy_score_train = list()
	for tweet in training_list_joy:
		joy_tweets_train.append(tweet.text);
		joy_score_train.append(float(tweet.intensity));
		total_dataset.append(tweet.text)

	joy_tweets_val = list();
	joy_score_val = list();
	for tweet in validation_list_joy:
		joy_tweets_val.append(tweet.text);
		joy_score_val.append(float(tweet.intensity));
		total_dataset.append(tweet.text)

	joy_tweets_test = list();
	joy_score_test = list();
	for tweet in test_list_joy:
		joy_tweets_test.append(tweet.text);
		joy_score_test.append(float(tweet.intensity));
		total_dataset.append(tweet.text)

	sadness_tweets_train = list()
	sadness_score_train = list()
	for tweet in training_list_sadness:
		sadness_tweets_train.append(tweet.text);
		sadness_score_train.append(float(tweet.intensity));
		total_dataset.append(tweet.text)

	sadness_tweets_val = list();
	sadness_score_val = list();
	for tweet in validation_list_sadness:
		sadness_tweets_val.append(tweet.text);
		sadness_score_val.append(float(tweet.intensity));
		total_dataset.append(tweet.text)

	sadness_tweets_test = list();
	sadness_score_test = list();
	for tweet in test_list_sadness:
		sadness_tweets_test.append(tweet.text);
		sadness_score_test.append(float(tweet.intensity));
		total_dataset.append(tweet.text)

	t = Tokenizer()
	t.fit_on_texts(total_dataset)
	word_index = t.word_index
	print(t.document_count)
	vocab_size = len(t.word_counts)
	print(vocab_size)
	print (len(word_index))
	max_len = 50

	anger_sequences_train = t.texts_to_sequences(anger_tweets_train)
	anger_sequences_val = t.texts_to_sequences(anger_tweets_val)
	anger_sequences_test = t.texts_to_sequences(anger_tweets_test)

	fear_sequences_train = t.texts_to_sequences(fear_tweets_train)
	fear_sequences_val = t.texts_to_sequences(fear_tweets_val)
	fear_sequences_test = t.texts_to_sequences(fear_tweets_test)

	joy_sequences_train = t.texts_to_sequences(joy_tweets_train)
	joy_sequences_val = t.texts_to_sequences(joy_tweets_val)
	joy_sequences_test = t.texts_to_sequences(joy_tweets_test)

	sadness_sequences_train = t.texts_to_sequences(sadness_tweets_train)
	sadness_sequences_val = t.texts_to_sequences(sadness_tweets_val)
	sadness_sequences_test = t.texts_to_sequences(sadness_tweets_test)
	
	EMBEDDING_DIM = 100
	GLOVE_DIR = "./Data/glove.twitter.27B/glove.twitter.27B.100d.txt"
	embeddings_index = {}
	f = open(GLOVE_DIR)
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    embeddings_index[word] = coefs
	f.close()
	print ('Read Glove and Made Dict')
	embedding_matrix = np.zeros((vocab_size + 1, EMBEDDING_DIM))
	number_found =0
	number_not_found = 0
	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None:
	        # words not found in embedding index will be all-zeros.
	        embedding_matrix[i] = embedding_vector
	        number_found+=1
	    else:
	    	#print (word)
	    	number_not_found+=1

	print(number_found)
	print(number_not_found)

	anger_padded_train 		= pad_sequences(anger_sequences_train,maxlen = max_len,padding = 'post')
	anger_padded_val 		= pad_sequences(anger_sequences_val,maxlen = max_len,padding = 'post')
	anger_padded_test		= pad_sequences(anger_sequences_test,maxlen = max_len,padding = 'post')

	fear_padded_train 	    = pad_sequences(fear_sequences_train,maxlen = max_len,padding = 'post')
	fear_padded_val 		= pad_sequences(fear_sequences_val,maxlen = max_len,padding = 'post')
	fear_padded_test		= pad_sequences(fear_sequences_test,maxlen = max_len,padding = 'post')
	
	joy_padded_train 	    = pad_sequences(joy_sequences_train,maxlen = max_len,padding = 'post')
	joy_padded_val 			= pad_sequences(joy_sequences_val,maxlen = max_len,padding = 'post')
	joy_padded_test			= pad_sequences(joy_sequences_test,maxlen = max_len,padding = 'post')
	
	sadness_padded_train    = pad_sequences(sadness_sequences_train,maxlen = max_len,padding = 'post')
	sadness_padded_val 		= pad_sequences(sadness_sequences_val,maxlen = max_len,padding = 'post')
	sadness_padded_test		= pad_sequences(sadness_sequences_test,maxlen = max_len,padding = 'post')

	anger_padded_train = np.array(anger_padded_train)
	fear_padded_train = np.array(fear_padded_train)
	joy_padded_train = np.array(joy_padded_train)
	sadness_padded_train = np.array(sadness_padded_train)
	anger_score_train = np.array(anger_score_train)
	fear_score_train = np.array(fear_score_train)
	joy_score_train = np.array(joy_score_train)
	sadness_score_train = np.array(sadness_score_train)

	anger_padded_val = np.array(anger_padded_val)
	fear_padded_val = np.array(fear_padded_val)
	joy_padded_val = np.array(joy_padded_val)
	sadness_padded_val = np.array(sadness_padded_val)
	anger_score_val = np.array(anger_score_val)
	fear_score_val = np.array(fear_score_val)
	joy_score_val = np.array(joy_score_val)
	sadness_score_val = np.array(sadness_score_val)

	anger_padded_test = np.array(anger_padded_test)
	fear_padded_test = np.array(fear_padded_test)
	joy_padded_test = np.array(joy_padded_test)
	sadness_padded_test = np.array(sadness_padded_test)
	anger_score_test = np.array(anger_score_test)
	fear_score_test = np.array(fear_score_test)
	joy_score_test = np.array(joy_score_test)
	sadness_score_test = np.array(sadness_score_test)

	TRAIN_LENGTH = 1147
	VAL_LENGTH = 110
	TEST_LENGTH = 995 

	def get_model():
		shared_lstm = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2, name='shared_lstm',return_sequences=True))
		
		anger_main_input = Input(shape=(max_len,), dtype='int32', name='anger_main_input')
		fear_main_input = Input(shape=(max_len,), dtype='int32', name='fear_main_input')
		joy_main_input = Input(shape=(max_len,), dtype='int32', name='joy_main_input')
		sadness_main_input = Input(shape=(max_len,), dtype='int32', name='sadness_main_input')

		anger_input = Embedding(len(word_index) + 1,
	                            EMBEDDING_DIM,weights=[embedding_matrix],
	                            input_length=max_len,
	                            trainable=False)(anger_main_input)
		encoded_anger = shared_lstm(anger_input)
		anger_lstm = (LSTM(64,dropout=0.2, recurrent_dropout=0.2, name='anger_lstm'))(encoded_anger)
		anger_dense = Dense(200,activation='sigmoid')(anger_lstm)
		anger_pred = Dense(1,activation = 'sigmoid')(anger_dense)

		fear_input = Embedding(len(word_index) + 1,
	                            EMBEDDING_DIM,weights=[embedding_matrix],
	                            input_length=max_len,
	                            trainable=False)(fear_main_input)
		encoded_fear = shared_lstm(fear_input)
		fear_lstm = (LSTM(64,dropout=0.2, recurrent_dropout=0.2, name='fear_lstm'))(encoded_fear)
		fear_dense = Dense(200,activation='sigmoid')(fear_lstm)
		fear_pred = Dense(1,activation = 'sigmoid')(fear_dense)

		joy_input = Embedding(len(word_index) + 1,
	                            EMBEDDING_DIM,weights=[embedding_matrix],
	                            input_length=max_len,
	                            trainable=False)(joy_main_input)
		encoded_joy = shared_lstm(joy_input)
		joy_lstm = (LSTM(64,dropout=0.2, recurrent_dropout=0.2, name='joy_lstm'))(encoded_joy)
		joy_dense = Dense(200,activation='sigmoid')(joy_lstm)
		joy_pred = Dense(1,activation = 'sigmoid')(joy_dense)

		sadness_input = Embedding(len(word_index) + 1,
	                            EMBEDDING_DIM,weights=[embedding_matrix],
	                            input_length=max_len,
	                            trainable=False)(sadness_main_input)
		encoded_sadness = shared_lstm(sadness_input)
		sadness_lstm = (LSTM(64,dropout=0.2, recurrent_dropout=0.2, name='sadness_lstm'))(encoded_sadness)
		sadness_dense = Dense(200,activation='sigmoid')(sadness_lstm)
		sadness_pred = Dense(1,activation = 'sigmoid')(sadness_dense)


		model = Model(inputs=[anger_main_input, fear_main_input,joy_main_input,sadness_main_input], outputs=[anger_pred,fear_pred,joy_pred,sadness_pred])
		model.compile(optimizer='adam', loss='mean_squared_error')
		print(model.summary())
		return model
	
	estimator = KerasRegressor(build_fn = get_model,epochs =50,batch_size=32,verbose=1)

	
	estimator.fit([anger_padded_train[:785],fear_padded_train[:785],joy_padded_train[:785],sadness_padded_train[:785]],[anger_score_train[:785],fear_score_train[:785],joy_score_train[:785],sadness_score_train[:785]],validation_data = ([anger_padded_val[:73],fear_padded_val[:73],joy_padded_val[:73],sadness_padded_val[:73]],[anger_score_val[:73],fear_score_val[:73],joy_score_val[:73],sadness_score_val[:73]]))
	
	train_prediction = estimator.predict([anger_padded_train[:785],fear_padded_train[:785],joy_padded_train[:785],sadness_padded_train[:785]])
	#print (len(train_prediction[0]))
	#print (len(train_prediction))
	print(i)
	print ('===TRAINING==== ')
	print(pearsonr(train_prediction[0],anger_score_train[:785]))
	print(spearmanr(train_prediction[0],anger_score_train[:785]))

	print(pearsonr(train_prediction[1],fear_score_train[:785]))
	print(spearmanr(train_prediction[1],fear_score_train[:785]))
	
	print(pearsonr(train_prediction[2],joy_score_train[:785]))
	print(spearmanr(train_prediction[2],joy_score_train[:785]))
	
	print(pearsonr(train_prediction[3],sadness_score_train[:785]))
	print(spearmanr(train_prediction[3],sadness_score_train[:785]))

	print(i)
	print ('===VALIDATION====')
	val_prediction = estimator.predict([anger_padded_val[:73],fear_padded_val[:73],joy_padded_val[:73],sadness_padded_val[:73]])
	#print (len(val_prediction[0]))
	#print (len(val_prediction))
	print(pearsonr(val_prediction[0],anger_score_val[:73]))
	print(spearmanr(val_prediction[0],anger_score_val[:73]))

	print(pearsonr(val_prediction[1],fear_score_val[:73]))
	print(spearmanr(val_prediction[1],fear_score_val[:73]))
	
	print(pearsonr(val_prediction[2],joy_score_val[:73]))
	print(spearmanr(val_prediction[2],joy_score_val[:73]))
	
	print(pearsonr(val_prediction[3],sadness_score_val[:73]))
	print(spearmanr(val_prediction[3],sadness_score_val[:73]))
	
	
	print ('===TESTING====')
	test_prediction = estimator.predict([anger_padded_test[:672],fear_padded_test[:672],joy_padded_test[:672],sadness_padded_test[:672]])
	print(pearsonr(test_prediction[0],anger_score_test[:672]))
	print(spearmanr(test_prediction[0],anger_score_test[:672]))

	print(pearsonr(test_prediction[1],fear_score_test[:672]))
	print(spearmanr(test_prediction[1],fear_score_test[:672]))
	
	print(pearsonr(test_prediction[2],joy_score_test[:672]))
	print(spearmanr(test_prediction[2],joy_score_test[:672]))
	
	print(pearsonr(test_prediction[3],sadness_score_test[:672]))
	print(spearmanr(test_prediction[3],sadness_score_test[:672]))




#---MAIN_FILE-------

training_filename_anger = "anger-ratings-0to1.train.txt"
training_list_anger = read_training_data(training_filename_anger);
validation_filename_anger = "anger-ratings-0to1.dev.gold.txt"
validation_list_anger = read_training_data(validation_filename_anger)
test_filename_anger = "anger-ratings-0to1.test.gold.txt"
test_list_anger = read_training_data(test_filename_anger)

training_filename_fear = "fear-ratings-0to1.train.txt"
training_list_fear = read_training_data(training_filename_fear);
validation_filename_fear = "fear-ratings-0to1.dev.gold.txt"
validation_list_fear = read_training_data(validation_filename_fear)
test_filename_fear = "fear-ratings-0to1.test.gold.txt"
test_list_fear = read_training_data(test_filename_fear)

training_filename_joy = "joy-ratings-0to1.train.txt"
training_list_joy = read_training_data(training_filename_joy);
validation_filename_joy = "joy-ratings-0to1.dev.gold.txt"
validation_list_joy = read_training_data(validation_filename_joy)
test_filename_joy = "joy-ratings-0to1.test.gold.txt"
test_list_joy = read_training_data(test_filename_joy)

training_filename_sadness = "sadness-ratings-0to1.train.txt"
training_list_sadness = read_training_data(training_filename_sadness);
validation_filename_sadness = "sadness-ratings-0to1.dev.gold.txt"
validation_list_sadness = read_training_data(validation_filename_sadness)
test_filename_sadness = "sadness-ratings-0to1.test.gold.txt"
test_list_sadness = read_training_data(test_filename_sadness)
print ("Done Reading")

lstm_model(training_list_anger,validation_list_anger,test_list_anger,training_list_fear,validation_list_fear,test_list_fear,training_list_joy,validation_list_joy,test_list_joy,training_list_sadness,validation_list_sadness,test_list_sadness)
