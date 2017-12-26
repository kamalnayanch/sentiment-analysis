
# Import Statements
# Import Basics
import numpy as np
import html
import re
from nltk import word_tokenize
from nltk import bigrams
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize.casual import TweetTokenizer
from keras.callbacks import ModelCheckpoint
import h5py

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
from keras.layers.convolutional import Conv1D,Conv2D
from keras.layers.pooling import MaxPooling1D,MaxPooling2D, GlobalMaxPooling1D, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM, GRU
from keras.preprocessing import sequence


#import of scipy

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
	split_string = \
	[word for word in string.split()
		if word not in stopwords.words('english')]

	return " ".join(split_string)


def embedding_cnn_glove(training_list,validation_list,test_list):
	
	tweets_train = list()
	score_train = list()
	total_dataset = list()
	for tweet in training_list:
		tweets_train.append(tweet.text);
		score_train.append(float(tweet.intensity));
		total_dataset.append(tweet.text)

	tweets_val = list();
	score_val = list();
	for tweet in validation_list:
		tweets_val.append(tweet.text);
		score_val.append(float(tweet.intensity));
		total_dataset.append(tweet.text)

	tweets_test = list();
	score_test = list();
	for tweet in test_list:
		tweets_test.append(tweet.text);
		score_test.append(float(tweet.intensity));
		total_dataset.append(tweet.text)

	
	t = Tokenizer()
	t.fit_on_texts(total_dataset)
	word_index = t.word_index
	print(t.document_count)
	vocab_size = len(t.word_counts)
	print(vocab_size)
	print (len(word_index))
	max_len = 50
	sequences_train = t.texts_to_sequences(tweets_train)
	# print (tweets_train[0])
	# print (sequences_train[0])
	# print (tweets_train[0][0:3])
	# print (word_index.get(tweets_train[0][0:3]))
	# print (word_index.get(sequences_train[0][0]))

	sequences_val = t.texts_to_sequences(tweets_val)
	sequences_test = t.texts_to_sequences(tweets_test)


	padded_train 	= pad_sequences(sequences_train,maxlen = max_len,padding = 'post')
	padded_val 		= pad_sequences(sequences_val,maxlen = max_len,padding = 'post')
	padded_test		= pad_sequences(sequences_test,maxlen = max_len,padding = 'post')


	EMBEDDING_DIM = 100
	X = np.ones((len(padded_train),max_len,EMBEDDING_DIM,1), dtype=np.int64) * -1
	y = np.array(score_train)

	X_val = np.ones((len(padded_val),max_len,EMBEDDING_DIM,1), dtype=np.int64) * -1
	y_val = np.array(score_val)

	X_test = np.ones((len(padded_test),max_len,EMBEDDING_DIM,1), dtype=np.int64) * -1
	y_test = np.array(score_test)
	print (len(y_val))
	print (len(X_val))
	print (len(y_test))
	print (len(X_test))
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
	    	print (word)
	    	number_not_found+=1

	print(number_found)
	print(number_not_found)

	for i in range(len(padded_train)):
		for j in range(max_len):
			X[i,j,:,0] = embedding_matrix[padded_train[i][j]]

	for i in range(len(padded_val)):
		for j in range(max_len):
			X_val[i,j,:,0] = embedding_matrix[padded_val[i][j]]
	
	for i in range(len(padded_test)):
		for j in range(max_len):
			X_test[i,j,:,0] = embedding_matrix[padded_test[i][j]]

	conv_1 = Conv1D(64, 5, activation='relu', name='conv1',input_shape=(max_len,))
	conv_2 = Conv1D(32, 3, activation='relu', name='conv2')
	conv_3 = Conv1D(32, 3, activation='relu', name='conv3')
	#pooling layers
	pool_1 = AveragePooling1D(pool_size=3, strides=2, name='pool1')
	pool_2 = AveragePooling1D(pool_size=3, strides=2, name='pool2')
	pool_3 = MaxPooling1D(pool_size=3, strides=2, name='pool3')
	pool_4 = MaxPooling1D(pool_size=3, strides=2, name='pool4')

	#LSTM Layers
	lstm_1 = LSTM(256, dropout=0.2, recurrent_dropout=0.2, name='lstm1', return_sequences=True)
	lstm_2 = LSTM(128, dropout=0.2, recurrent_dropout=0.2, name='lstm2', return_sequences=True)
	lstm_3 = LSTM(64, dropout=0.2, recurrent_dropout=0.2, name='lstm3')
	lstm_4 = LSTM(32, dropout=0.2, recurrent_dropout=0.2, name='lstm4',return_sequences = True)
	#GRU Layers

	gru_1 = GRU(256, dropout=0.2, recurrent_dropout=0.2, name='gru1', return_sequences=True)
	gru_2 = GRU(128, dropout=0.2, recurrent_dropout=0.2, name='gru2', return_sequences=True)
	gru_3 = GRU(64, dropout=0.2, recurrent_dropout=0.2, name='gru3')

	#Bidirectional Layers

	bi_lstm_1 = Bidirectional(lstm_1);
	bi_lstm_2 = Bidirectional(lstm_2);
	bi_lstm_3 = Bidirectional(lstm_3);
	bi_lstm_4 = Bidirectional(lstm_4);

	#Dense layers
	dense_1 = Dense(200, activation='relu', name='dense1')
	dense_2 = Dense(1, activation='sigmoid', name='dense2')

	def get_model():
		model = Sequential()
		model.add(conv_1)
		model.add(Dropout(0.3))
		model.add(pool_3)
		model.add(conv_2)
		model.add(Dropout(0.3))
		model.add(pool_3)
		# model.add(conv_3)
		# model.add(Dropout(0.3))	
		# model.add(pool_3)

		model.add(Flatten())
		#model.add(Dense(200, activation='relu', name='dense3'))
		model.add(dense_1)
		model.add(dense_2)
		#compile the model
		model.compile(optimizer='adam', loss='mean_squared_error')
		# summarize the model
		print(model.summary())
		# fit the model
		return model


	estimator = KerasRegressor(build_fn = get_model,epochs =50,batch_size=32,verbose=1)
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
	

training_filename = "anger-ratings-0to1.train.txt"
training_list = read_training_data(training_filename);
validation_filename = "anger-ratings-0to1.dev.gold.txt"
validation_list = read_training_data(validation_filename)
test_filename = "anger-ratings-0to1.test.gold.txt"
test_list = read_training_data(test_filename)
print ("Done Reading")
# write_training_file(training_list,'preprocessed'+training_filename)
# write_training_file(validation_list,'preprocessed'+validation_filename)
# write_training_file(test_list,'preprocessed'+test_filename)
# print("Done Writing")
embedding_cnn_glove(training_list,validation_list,test_list)
