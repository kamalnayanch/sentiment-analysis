import numpy as np
import html
import re
from nltk import word_tokenize
from nltk import bigrams
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize.casual import TweetTokenizer
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
	string = re.sub(r"#", "", string) 						 # remove mentions of hashtags	
	string = re.sub(r"\*", "", string)
	string = re.sub(r"\'s", "", string)
	# change 'm to am ,'ve to have, n't to not, 're to are, 'd to would,'ll to will
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
	string = re.sub(r'[^\x00-\x7F]',' ', string) # remove non-ascii character mentions
	string = re.sub(r"\s{2,}", " ", string)	#remove multiple mentions of space with single
	string = string.rstrip(',|.|;|:|\'|\"') 
	string = string.lstrip('\'|\"')

	return string.lower() # remove_stopwords(string.strip().lower())
	
# function to remove stop words
def remove_stopwords(string):
	split_string = \
	[word for word in string.split()
		if word not in stopwords.words('english')]

	return " ".join(split_string)


# Deep learning Method on top of tf-idf embeddings

def embedding_lstm_tokenizer(training_list,validation_list,test_list):
	
	tweets_train = list();
	score_train = list();
	for tweet in training_list:
		tweets_train.append(tweet.text);
		score_train.append(float(tweet.intensity));

	tweets_val = list();
	score_val = list();
	for tweet in validation_list:
		tweets_val.append(tweet.text);
		score_val.append(float(tweet.intensity));

	tweets_test = list();
	score_test = list();
	for tweet in test_list:
		tweets_test.append(tweet.text);
		score_test.append(float(tweet.intensity));
	
	t = Tokenizer()
	t.fit_on_texts(tweets_train)
	print(t.document_count)
	vocab_size = len(t.word_counts)
	print(vocab_size)
	encoded_tweets_train = t.texts_to_matrix(tweets_train, mode='tfidf')
	encoded_tweets_val = t.texts_to_matrix(tweets_val,mode = 'tfidf')
	encoded_tweets_test = t.texts_to_matrix(tweets_test,mode = 'tfidf')

	#Embedding Layes
	embedding_1 = Embedding(vocab_size,50, input_length=vocab_size+1)
	#LSTM Layers
	lstm_1 = LSTM(256, dropout=0.2, recurrent_dropout=0.2, name='lstm1', return_sequences=True)
	lstm_2 = LSTM(128, dropout=0.2, recurrent_dropout=0.2, name='lstm2', return_sequences=True)
	lstm_3 = LSTM(32, dropout=0.2, recurrent_dropout=0.2, name='lstm3')
	#Dense Layers
	dense_1 = Dense(50, activation='relu', name='dense1')
	dense_2 = Dense(1, activation='sigmoid', name='dense2')

	def get_model():
		model = Sequential()
		model.add(embedding_1)
		model.add(lstm_3)
		#model.add(Flatten())
		model.add(dense_1)
		model.add(dense_2)
		#compile the model
		model.compile(optimizer='adam', loss='mean_squared_error')
		# summarize the model
		print(model.summary())
		# fit the model
		return model

	#create the model
	
	estimator = KerasRegressor(build_fn = get_model,epochs =4,batch_size=32,verbose=1)
	estimator.fit(encoded_tweets_train,score_train)

	train_prediction = estimator.predict(encoded_tweets_train)
	print(pearsonr(train_prediction,score_train))
	print(spearmanr(train_prediction,score_train))

	val_prediction = estimator.predict(encoded_tweets_val)
	print(pearsonr(val_prediction,score_val))
	print(spearmanr(val_prediction,score_val))
	
	test_prediction = estimator.predict(encoded_tweets_test)
	print(pearsonr(test_prediction,score_test))
	print(spearmanr(test_prediction,score_test))


def embedding_lstm_one_hot_vector(training_list,validation_list,test_list):
	
	tweets_train = list();
	score_train = list();
	for tweet in training_list:
		tweets_train.append(tweet.text);
		score_train.append(float(tweet.intensity));

	tweets_val = list();
	score_val = list();
	for tweet in validation_list:
		tweets_val.append(tweet.text);
		score_val.append(float(tweet.intensity));

	tweets_test = list();
	score_test = list();
	for tweet in test_list:
		tweets_test.append(tweet.text);
		score_test.append(float(tweet.intensity));

	#Assumption: Vocab_size = 5000, Max Sequence Length = 50
	vocab_size = 5000
	max_len = 30

	#Encoding using one hot vector 
	encoded_tweets_train	= [one_hot(d, vocab_size) for d in tweets_train]
	encoded_tweets_val		= [one_hot(d, vocab_size) for d in tweets_val]
	encoded_tweets_test		= [one_hot(d, vocab_size) for d in tweets_test]

	#Padding all the sequences so as to make them of equal length

	padded_train 	= pad_sequences(encoded_tweets_train,maxlen = max_len,padding = 'post')
	padded_val 		= pad_sequences(encoded_tweets_val,maxlen = max_len,padding = 'post')
	padded_test		= pad_sequences(encoded_tweets_test,maxlen = max_len,padding = 'post')

	#Embedding Layes
	embedding_1 = Embedding(vocab_size,50, input_length=max_len)

	#conv layers
	conv_1 = Conv1D(128, 3, activation='relu', name='conv1')
	conv_2 = Conv1D(128, 3, activation='relu', name='conv2')

	#pooling layers
	pool_3 = MaxPooling1D(pool_size=3, strides=2, name='pool3')
	pool_4 = MaxPooling1D(pool_size=3, strides=2, name='pool4')

	#LSTM Layers
	lstm_1 = LSTM(256, dropout=0.2, recurrent_dropout=0.2, name='lstm1', return_sequences=True)
	lstm_2 = LSTM(128, dropout=0.2, recurrent_dropout=0.2, name='lstm2', return_sequences=True)
	lstm_3 = LSTM(64, dropout=0.2, recurrent_dropout=0.2, name='lstm3')
	lstm_4 = LSTM(32, dropout=0.2, recurrent_dropout=0.2, name='lstm4',return_sequences = True)
	#Dense Layers
	dense_1 = Dense(200, activation='relu', name='dense1')
	dense_2 = Dense(1, activation='sigmoid', name='dense2')

	def get_model():
		model = Sequential()
		model.add(embedding_1)
		#model.add(bi_lstm_1)
		model.add(conv_1)
		model.add(pool_3)
		model.add(conv_2)
		model.add(pool_4)
		#model.add(bi_lstm_3)
		#model.add(gru_2)
		#model.add(gru_3)
		#model.add(Flatten())
		#model.add(lstm_1)
		model.add(lstm_2)
		model.add(lstm_3)
		model.add(dense_1)
		model.add(dense_2)
		#compile the model
		model.compile(optimizer='adam', loss='mean_squared_error')
		# summarize the model
		print(model.summary())
		# fit the model
		return model

	#create the model
	
	estimator = KerasRegressor(build_fn = get_model,epochs = 30,batch_size=32,verbose=1)
	estimator.fit(padded_train,score_train,validation_data = (padded_val,score_val))
	train_prediction = estimator.predict(padded_train)
	print(pearsonr(train_prediction,score_train))
	print(spearmanr(train_prediction,score_train))

	val_prediction = estimator.predict(padded_val)
	print(pearsonr(val_prediction,score_val))
	print(spearmanr(val_prediction,score_val))
	
	test_prediction = estimator.predict(padded_test)
	print(pearsonr(test_prediction,score_test))
	print(spearmanr(test_prediction,score_test))
	


def embedding_lstm_glove(input_epochs,training_list,validation_list,test_list):
	
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
	# #If we have glove in dictionary this code could be used 
	# for word, i in word_index.items():
	#     embedding_vector = glove_train.item().get(word)
	#     if embedding_vector is not None:
	#         # words not found in embedding index will be all-zeros.
	#         embedding_matrix[i] = embedding_vector
	#         number_found +=1
	#         continue
	#         #print (hi)
	#     embedding_vector = glove_val.item().get(word)
	#     if embedding_vector is not None:
	#         # words not found in embedding index will be all-zeros.
	#         embedding_matrix[i] = embedding_vector
	#         number_found +=1
	#         continue
	#     embedding_vector = glove_test.item().get(word)
	#     if embedding_vector is not None:
	#         # words not found in embedding index will be all-zeros.
	#         embedding_matrix[i] = embedding_vector
	#         number_found +=1
	#         continue
	#     number_not_found +=1
	#     # print('Not found')
	#     #print(word)

	#     #print(embedding_matrix[i])

	# print(number_found)
	# print(number_not_found)

	padded_train 	= pad_sequences(sequences_train,maxlen = max_len,padding = 'post')
	padded_val 		= pad_sequences(sequences_val,maxlen = max_len,padding = 'post')
	padded_test		= pad_sequences(sequences_test,maxlen = max_len,padding = 'post')
	#print(padded_train)
	#Embedding Layes
	embedding_1 = Embedding(len(word_index) + 1,
	                            EMBEDDING_DIM,weights=[embedding_matrix],
	                            input_length=max_len,
	                            trainable=False)

	#conv layers
	conv_1 = Conv1D(128, 3, activation='relu', name='conv1')
	conv_2 = Conv1D(64, 3, activation='relu', name='conv2')
	conv_3 = Conv1D(32, 3, activation='relu', name='conv3')

	#pooling layers
	pool_3 = MaxPooling1D(pool_size=3, strides=2, name='pool3')
	pool_4 = MaxPooling1D(pool_size=3, strides=2, name='pool4')

	#LSTM Layers
	lstm_1 = LSTM(256, dropout=0.2, recurrent_dropout=0.2, name='lstm1', return_sequences=True)
	lstm_2 = LSTM(128, dropout=0.2, recurrent_dropout=0.2, name='lstm2', return_sequences=True)
	lstm_3 = LSTM(64, dropout=0.2, recurrent_dropout=0.2, name='lstm3')
	lstm_5 = LSTM(64, dropout=0.2, recurrent_dropout=0.2, name='lstm5',return_sequences = True)
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

	#Dense Layers
	dense_1 = Dense(200, activation='relu', name='dense1')
	dense_2 = Dense(1, activation='sigmoid', name='dense2')

	def get_model():
		model = Sequential()
		model.add(embedding_1)
		model.add(conv_1)
		model.add(pool_3)
		model.add(conv_2)
		model.add(pool_4)
		#model.add(bi_lstm_1)
		# model.add(conv_1)
		# model.add(Dropout(0.3))
		# model.add(pool_4)
		# model.add(conv_2)
		# model.add(Dropout(0.3))
		# #model.add(pool_4)
		# model.add(conv_3)
		# model.add(Dropout(0.3))
		# model.add(pool_4)
		#model.add(lstm_1)
		#model.add(bi_lstm_2)
		model.add(lstm_5)
		model.add(lstm_3)
		#model.add(lstm_3)
		#model.add(lstm_2)
		#model.add(lstm_3)
		#model.add(Flatten())
		model.add(dense_1)
		model.add(dense_2)
		#compile the model
		model.compile(optimizer='adam', loss='mean_squared_error')
		# summarize the model
		print(model.summary())
		# fit the model
		return model

	#create the model
	
	estimator = KerasRegressor(build_fn = get_model,epochs =input_epochs,batch_size=32,verbose=1)
	estimator.fit(padded_train,score_train,validation_data = (padded_val,score_val))
	train_prediction = estimator.predict(padded_train)
	print(pearsonr(train_prediction,score_train))
	print(spearmanr(train_prediction,score_train))

	val_prediction = estimator.predict(padded_val)
	print(pearsonr(val_prediction,score_val))
	print(spearmanr(val_prediction,score_val))
	
	test_prediction = estimator.predict(padded_test)
	print(pearsonr(test_prediction,score_test))
	print(spearmanr(test_prediction,score_test))

	# #For building the submission file
	# thefile = open('submission_50', 'w')
	# for item in test_prediction:
	# 	thefile.write("%f\n" % (item))

	# thefile.close()

# def embedding_lstm_glove_lexicons(input_epochs,training_list,validation_list,test_list):
	
# 	tweets_train = list()
# 	score_train = list()
# 	total_dataset = list()
# 	for tweet in training_list:
# 		tweets_train.append(tweet.text);
# 		score_train.append(float(tweet.intensity));
# 		total_dataset.append(tweet.text)

# 	tweets_val = list();
# 	score_val = list();
# 	for tweet in validation_list:
# 		tweets_val.append(tweet.text);
# 		score_val.append(float(tweet.intensity));
# 		total_dataset.append(tweet.text)

# 	tweets_test = list();
# 	score_test = list();
# 	for tweet in test_list:
# 		tweets_test.append(tweet.text);
# 		score_test.append(float(tweet.intensity));
# 		total_dataset.append(tweet.text)

	
# 	t = Tokenizer()
# 	t.fit_on_texts(total_dataset)
# 	word_index = t.word_index
# 	print(t.document_count)
# 	vocab_size = len(t.word_counts)
# 	print(vocab_size)
# 	print (len(word_index))
# 	max_len = 50
# 	sequences_train = t.texts_to_sequences(tweets_train)
# 	# print (tweets_train[0])
# 	# print (sequences_train[0])
# 	# print (tweets_train[0][0:3])
# 	# print (word_index.get(tweets_train[0][0:3]))
# 	# print (word_index.get(sequences_train[0][0]))

# 	sequences_val = t.texts_to_sequences(tweets_val)
# 	sequences_test = t.texts_to_sequences(tweets_test)

# 	EMBEDDING_DIM = 109
# 	GLOVE_DIR = "./Data/glove.twitter.27B/glove.twitter.27B.100d.txt"
# 	embeddings_index = {}
# 	f = open(GLOVE_DIR)
# 	for line in f:
# 	    values = line.split()
# 	    word = values[0]
# 	    coefs = np.asarray(values[1:], dtype='float32')
# 	    embeddings_index[word] = coefs
# 	f.close()
# 	print ('Read Glove and Made Dict')
	

# 	dict1_anger = np.load("./Data/dict1_anger.npy").item()
# 	dict1_fear = np.load("./Data/dict1_fear.npy").item()
# 	dict1_joy = np.load("./Data/dict1_joy.npy").item()
# 	dict1_sadness = np.load("./Data/dict1_sadness.npy").item()

# 	dict2_anger = np.load("./Data/dict2_anger.npy").item()
# 	dict2_fear = np.load("./Data/dict2_fear.npy").item()
# 	dict2_joy = np.load("./Data/dict2_joy.npy").item()
# 	dict2_sadness = np.load("./Data/dict2_sadness.npy").item()

# 	dict3 = np.load("./Data/dict3.npy").item()

# 	embedding_matrix = np.zeros((vocab_size + 1, EMBEDDING_DIM))
# 	number_found =0
# 	number_not_found = 0
# 	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
# 	for word, i in word_index.items():
	    
# 	    embedding_vector = embeddings_index.get(word)
# 	    lex_anger_1 = dict1_anger.get(word)
# 	    lex_fear_1 = dict1_fear.get(word)
# 	    lex_joy_1 = dict1_joy.get(word)
# 	    lex_sadness_1 = dict1_sadness.get(word)

# 	    lex_anger_2 = dict2_anger.get(word)
# 	    lex_fear_2 = dict2_fear.get(word)
# 	    lex_joy_2 = dict2_joy.get(word)
# 	    lex_sadness_2 = dict2_sadness.get(word)
# 	    lex_3 = dict3.get(word)

# 	    if embedding_vector is not None:
# 	        # words not found in embedding index will be all-zeros.
# 	        embedding_matrix[i][:100] = embedding_vector
# 	        number_found+=1
# 	    else:
# 	    	#print (word)
# 	    	number_not_found+=1
# 	    if lex_anger_1 is not None:
# 	    	embedding_matrix[i][100] = lex_anger_1
# 	    if lex_fear_1 is not None:
# 	    	embedding_matrix[i][101] = lex_fear_1
# 	    if lex_joy_1 is not None:
# 	    	embedding_matrix[i][102] = lex_joy_1
# 	    if lex_sadness_1 is not None:
# 	    	embedding_matrix[i][103] = lex_sadness_1
# 	    if lex_anger_2 is not None:
# 	    	embedding_matrix[i][104] = lex_anger_2
# 	    if lex_fear_2 is not None:
# 	    	embedding_matrix[i][105] = lex_fear_2
# 	    if lex_joy_2 is not None:
# 	    	embedding_matrix[i][106] = lex_joy_2
# 	    if lex_sadness_2 is not None:
# 	    	embedding_matrix[i][107] = lex_sadness_2
# 	    if lex_3 is not None:
# 	    	embedding_matrix[i][108] = lex_3

# 	# #If we have glove in dictionary this code could be used 
# 	# for word, i in word_index.items():
# 	#     embedding_vector = glove_train.item().get(word)
# 	#     if embedding_vector is not None:
# 	#         # words not found in embedding index will be all-zeros.
# 	#         embedding_matrix[i] = embedding_vector
# 	#         number_found +=1
# 	#         continue
# 	#         #print (hi)
# 	#     embedding_vector = glove_val.item().get(word)
# 	#     if embedding_vector is not None:
# 	#         # words not found in embedding index will be all-zeros.
# 	#         embedding_matrix[i] = embedding_vector
# 	#         number_found +=1
# 	#         continue
# 	#     embedding_vector = glove_test.item().get(word)
# 	#     if embedding_vector is not None:
# 	#         # words not found in embedding index will be all-zeros.
# 	#         embedding_matrix[i] = embedding_vector
# 	#         number_found +=1
# 	#         continue
# 	#     number_not_found +=1
# 	#     # print('Not found')
# 	#     #print(word)

# 	#     #print(embedding_matrix[i])

# 	# print(number_found)
# 	# print(number_not_found)

# 	padded_train 	= pad_sequences(sequences_train,maxlen = max_len,padding = 'post')
# 	padded_val 		= pad_sequences(sequences_val,maxlen = max_len,padding = 'post')
# 	padded_test		= pad_sequences(sequences_test,maxlen = max_len,padding = 'post')
# 	#print(padded_train)
# 	#Embedding Layes
# 	embedding_1 = Embedding(len(word_index) + 1,
# 	                            EMBEDDING_DIM,weights=[embedding_matrix],
# 	                            input_length=max_len,
# 	                            trainable=False)

# 	#conv layers
# 	conv_1 = Conv1D(128, 3, activation='relu', name='conv1')
# 	conv_2 = Conv1D(64, 3, activation='relu', name='conv2')
# 	conv_3 = Conv1D(32, 3, activation='relu', name='conv3')

# 	#pooling layers
# 	pool_3 = MaxPooling1D(pool_size=3, strides=2, name='pool3')
# 	pool_4 = MaxPooling1D(pool_size=3, strides=2, name='pool4')

# 	#LSTM Layers
# 	lstm_1 = LSTM(256, dropout=0.2, recurrent_dropout=0.2, name='lstm1', return_sequences=True)
# 	lstm_2 = LSTM(128, dropout=0.2, recurrent_dropout=0.2, name='lstm2', return_sequences=True)
# 	lstm_3 = LSTM(64, dropout=0.2, recurrent_dropout=0.2, name='lstm3')
# 	lstm_5 = LSTM(64, dropout=0.2, recurrent_dropout=0.2, name='lstm5',return_sequences = True)
# 	lstm_4 = LSTM(32, dropout=0.2, recurrent_dropout=0.2, name='lstm4',return_sequences = True)
# 	#GRU Layers

# 	gru_1 = GRU(256, dropout=0.2, recurrent_dropout=0.2, name='gru1', return_sequences=True)
# 	gru_2 = GRU(128, dropout=0.2, recurrent_dropout=0.2, name='gru2', return_sequences=True)
# 	gru_3 = GRU(64, dropout=0.2, recurrent_dropout=0.2, name='gru3')

# 	#Bidirectional Layers

# 	bi_lstm_1 = Bidirectional(lstm_1);
# 	bi_lstm_2 = Bidirectional(lstm_2);
# 	bi_lstm_3 = Bidirectional(lstm_3);
# 	bi_lstm_4 = Bidirectional(lstm_4);

# 	#Dense Layers
# 	dense_1 = Dense(200, activation='relu', name='dense1')
# 	dense_2 = Dense(1, activation='sigmoid', name='dense2')

# 	def get_model():
# 		model = Sequential()
# 		model.add(embedding_1)
# 		# model.add(conv_1)
# 		# model.add(pool_3)
# 		# model.add(conv_2)
# 		# model.add(pool_4)
# 		#model.add(bi_lstm_1)
# 		# model.add(conv_1)
# 		# model.add(Dropout(0.3))
# 		# model.add(pool_4)
# 		# model.add(conv_2)
# 		# model.add(Dropout(0.3))
# 		# #model.add(pool_4)
# 		# model.add(conv_3)
# 		# model.add(Dropout(0.3))
# 		# model.add(pool_4)
# 		#model.add(lstm_1)
# 		#model.add(bi_lstm_2)
# 		model.add(lstm_5)
# 		model.add(lstm_3)
# 		#model.add(lstm_3)
# 		#model.add(lstm_2)
# 		#model.add(lstm_3)
# 		#model.add(Flatten())
# 		model.add(dense_1)
# 		model.add(dense_2)
# 		#compile the model
# 		model.compile(optimizer='adam', loss='mean_squared_error')
# 		# summarize the model
# 		print(model.summary())
# 		# fit the model
# 		return model

# 	#create the model
	
# 	estimator = KerasRegressor(build_fn = get_model,epochs =input_epochs,batch_size=32,verbose=1)
# 	estimator.fit(padded_train,score_train,validation_data = (padded_val,score_val))
# 	train_prediction = estimator.predict(padded_train)
# 	print(pearsonr(train_prediction,score_train))
# 	print(spearmanr(train_prediction,score_train))

# 	val_prediction = estimator.predict(padded_val)
# 	print(pearsonr(val_prediction,score_val))
# 	print(spearmanr(val_prediction,score_val))
	
# 	test_prediction = estimator.predict(padded_test)
# 	print(pearsonr(test_prediction,score_test))
# 	print(spearmanr(test_prediction,score_test))

# 	# #For building the submission file
# 	# thefile = open('submission_50', 'w')
# 	# for item in test_prediction:
# 	# 	thefile.write("%f\n" % (item))

# 	# thefile.close()
		


#---MAIN FUNCTION---

#Read all the files
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

print ("Done Reading the input files")

#----- For Using LSTM on top of Tf-idf 
# print('Training A LSTM on top of TF-IDF encoding')
# embedding_lstm_tokenizer(training_list_anger,validation_list_anger,test_list_anger)
# embedding_lstm_tokenizer(training_list_fear,validation_list_fear,test_list_fear)
# embedding_lstm_tokenizer(training_list_joy,validation_list_joy,test_list_joy)
# embedding_lstm_tokenizer(training_list_sadness,validation_list_sadness,test_list_sadness)
# exit()


#----- For Using LSTM on top of one-hot vector encoding 
# print('Training A LSTM on top of one-hot vector encoding')
# embedding_lstm_one_hot_vector(training_list_anger,validation_list_anger,test_list_anger)
# embedding_lstm_one_hot_vector(training_list_fear,validation_list_fear,test_list_fear)
# embedding_lstm_one_hot_vector(training_list_joy,validation_list_joy,test_list_joy)
# embedding_lstm_one_hot_vector(training_list_sadness,validation_list_sadness,test_list_sadness)
# exit()

# print('Anger')
# embedding_lstm_glove(30,training_list_anger,validation_list_anger,test_list_anger)
# print('Fear')
# embedding_lstm_glove(30,training_list_fear,validation_list_fear,test_list_fear)
# print('Joy')
# embedding_lstm_glove(30,training_list_joy,validation_list_joy,test_list_joy)
# print('Sadness')
# embedding_lstm_glove(30,training_list_sadness,validation_list_sadness,test_list_sadness)


# print('Anger')
# embedding_lstm_glove_lexicons(50,training_list_anger,validation_list_anger,test_list_anger)
# print('Fear')
# embedding_lstm_glove_lexicons(50,training_list_fear,validation_list_fear,test_list_fear)
# print('Joy')
# embedding_lstm_glove_lexicons(50,training_list_joy,validation_list_joy,test_list_joy)
# print('Sadness')
# embedding_lstm_glove_lexicons(50,training_list_sadness,validation_list_sadness,test_list_sadness)

# print('Anger')
# embedding_lstm_glove_lexicons(100,training_list_anger,validation_list_anger,test_list_anger)
# print('Fear')
# embedding_lstm_glove_lexicons(100,training_list_fear,validation_list_fear,test_list_fear)
# print('Joy')
# embedding_lstm_glove_lexicons(100,training_list_joy,validation_list_joy,test_list_joy)
# print('Sadness')
# embedding_lstm_glove_lexicons(100,training_list_sadness,validation_list_sadness,test_list_sadness)

# print('Anger')
# embedding_lstm_glove_lexicons(150,training_list_anger,validation_list_anger,test_list_anger)
# print('Fear')
# embedding_lstm_glove_lexicons(150,training_list_fear,validation_list_fear,test_list_fear)
# print('Joy')
# embedding_lstm_glove_lexicons(150,training_list_joy,validation_list_joy,test_list_joy)
# print('Sadness')
# embedding_lstm_glove_lexicons(150,training_list_sadness,validation_list_sadness,test_list_sadness)

# print('Anger')
# embedding_lstm_glove_lexicons(200,training_list_anger,validation_list_anger,test_list_anger)
# print('Fear')
# embedding_lstm_glove_lexicons(200,training_list_fear,validation_list_fear,test_list_fear)
# print('Joy')
# embedding_lstm_glove_lexicons(200,training_list_joy,validation_list_joy,test_list_joy)
# print('Sadness')
# embedding_lstm_glove_lexicons(200,training_list_sadness,validation_list_sadness,test_list_sadness)
# exit()


print('Anger')
embedding_lstm_glove(150,training_list_anger,validation_list_anger,test_list_anger)
print('Fear')
embedding_lstm_glove(150,training_list_fear,validation_list_fear,test_list_fear)
print('Joy')
embedding_lstm_glove(150,training_list_joy,validation_list_joy,test_list_joy)
print('Sadness')
embedding_lstm_glove(150,training_list_sadness,validation_list_sadness,test_list_sadness)
#embedding_lstm_glove_new(training_list_anger,validation_list_anger,test_list_anger,training_list_fear,validation_list_fear,test_list_fear,training_list_joy,validation_list_joy,test_list_joy,training_list_sadness,validation_list_sadness,test_list_sadness)
# write_training_file(training_list,'preprocessed'+training_filename)
# write_training_file(validation_list,'preprocessed'+validation_filename)
# write_training_file(test_list,'preprocessed'+test_filename)
# print("Done Writing")
#embedding_lstm_glove(training_list,validation_list,test_list)





