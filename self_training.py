"""
There are two domains" hotel and 
"""


import numpy as np
np.random.seed(1337) 

import tensorflow as tf


import keras

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, concatenate, Merge
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, LSTM, Concatenate, Input
from keras.models import Model
from keras import backend as K

from sklearn.metrics import (precision_score, recall_score,
f1_score, accuracy_score, confusion_matrix, roc_auc_score, auc)
import sklearn.metrics as sklm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle 
from keras.engine.topology import Layer
from keras import backend as K, initializers, regularizers, constraints


import pickle
import os
import math 

# Set this variable to toggle between hotels and electronics dataset for training and testing.
data_genre = "electronics"

maxSeqLength = 106
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 106

# set parameters:

batch_size = 64
embedding_dims = 300

filters = 250
kernel_size = 5
hidden_dims = 250
epochs = 20

# Each element of the data pickle is a sequence of word vectors. It is entire dataset: concatenation of (train, val, test)  
data_pickle = "./extracted/{}_data_labeled.p".format(data_genre)

labels_train = pickle.load(open("./extracted/{}_train_labels.p".format(data_genre), "rb"))
labels_valid = pickle.load(open("./extracted/{}_valid_labels.p".format(data_genre), "rb"))
labels_test = pickle.load(open("./extracted/{}_test_labels.p".format(data_genre), "rb"))

# Loading the linguistic features which have already been saved as pickle files.
x_train_ling  = pickle.load(open('./Data/Linguistic_features/{}/{}_train_features.p'.format(data_genre, data_genre), "rb"), encoding = 'latin1')
x_val_ling = pickle.load(open('./Data/Linguistic_features/{}/{}_valid_features.p'.format(data_genre, data_genre), "rb"), encoding = 'latin1')
x_test_ling = pickle.load(open('./Data/Linguistic_features/{}/{}_test_features.p'.format(data_genre, data_genre), "rb"), encoding = 'latin1')
data_ling = np.vstack((x_train_ling, x_val_ling, x_test_ling))

# The embedding matrix pickle. The embedding matrix is a 2D matrix of dimension Vocabulary_size x word_vector_dimension
embedding_matrix_pickle = "./extracted/{}_embedding_matrix.p".format(data_genre)

data = pickle.load(open(data_pickle, "rb"))

# Loading the labels, and converting them to one hot vectors of dimension 2.
labels = labels_train + labels_valid + labels_test
labels = to_categorical(labels, num_classes = 2)

print('Shape of data tensor:', data.shape)
print('Shape of ling features tensor:',data_ling.shape)

print('Shape of label tensor:', labels.shape)


print('Loading embedding matrix.')
embedding_matrix = pickle.load(open(embedding_matrix_pickle, "rb"))
num_words = embedding_matrix.shape[0] -1

skf = StratifiedKFold(n_splits=5)
folds = 0

# test scores for each iteration in the 5 folds
precision_kfold = []
recall_kfold = []
f1_kfold = []
train_loss_kfold = []

# validation scores for each iteration in the 5 folds
f1_val_kfold = []
precision_val_kfold = []
recall_val_kfold = []
acc_val_kfold = []
mae_val_kfold  = []
loss_val_kfold = []

def print_best_performance(values_matrix, results, strategy):
	precision_kfold, recall_kfold, f1_kfold = results

	final_precision = []
	final_recall = []
	final_f1 = []

	for i in range(5):
		if(strategy == 'max'):
			index = np.argmax(values_matrix[i])
		else:
			index = np.argmin(values_matrix[i])
		final_precision.append(precision_kfold[i][index])
		final_recall.append(recall_kfold[i][index])
		final_f1.append(f1_kfold[i][index])

	print('Recall: {0:.3f}'.format(np.mean(final_recall)))
	print('Precision: {0:.3f}'.format(np.mean(final_precision)))
	print('F1: {0:.3f}'.format(np.mean(final_f1)))	


# Defining the dot_product method. To be used by the Attention class

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        # todo: check that this is correct
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


# The Attention class. Code courtesy: 
class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False,
                 **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Note: The layer has been tested with Keras 1.x
        Example:
        
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2 - Get the attention scores
            hidden = LSTM(64, return_sequences=True)(words)
            sentence, word_scores = Attention(return_attention=True)(hidden)
        """
        self.supports_masking = True
        self.return_attention = return_attention
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = x * K.expand_dims(a)

        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]





# Maximum number of iterations of the self training
no_iter = 10

# Iterating over all the folds
for dev_index,  test_index in skf.split(data, labels[:,1]):
	folds +=1
	print("*****  Fold {} ******".format(folds))

	# Load the unlabeled data. The data is a 3D tensor of dimension no_of_sentences * max_sequence_length * word_vector_size 

	unlabeled_data = pickle.load(open("./extracted/{}_data_unlabeled.p".format(data_genre), "rb"))

	print("Length of unlabeled data is {}".format(len(unlabeled_data)))

	# Divide the dataset into train, valid and test
	x_dev, x_test = data[dev_index], data[test_index]
	y_dev, y_test = labels[dev_index], labels[test_index]
	x_dev_ling = data_ling[dev_index]
	x_test_ling = data_ling[test_index]

	dev_indices = range(len(x_dev))

	x_train_indices, x_val_indices, y_train, y_val = train_test_split(dev_indices, y_dev, test_size = 0.1, stratify = y_dev)
	x_train = x_dev[x_train_indices]
	x_val = x_dev[x_val_indices]

	x_train_ling = x_dev_ling[x_train_indices]
	x_val_ling = x_dev_ling[x_val_indices]

	unlabeled_ling =  pickle.load(open('./Data/Linguistic_features/{}/{}_unlabeled_features.p'.format(data_genre, data_genre), "rb"), encoding = 'latin1')
	print('Shape of Unlabeled data tensor:', unlabeled_data.shape)
	print('Shape of Unlabeled ling features tensor:',unlabeled_ling.shape)
	
	print('Shape of data train tensor:', x_train.shape)
	print('Shape of ling features train tensor:',x_train_ling.shape)

	# This stores the metrics on the test set after each iteration
	precision_kiter = []
	recall_kiter = []
	f1_kiter = []
	train_loss_kiter = []

	# This stores the metrics on the validation set after each iteration
	f1_val_kiter = []
	precision_val_kiter = []
	recall_val_kiter = []
	acc_val_kiter = []
	mae_val_kiter  = []
	loss_val_kiter = []

	early_stopping_iter =False
	best_model_loss = math.inf
	best_iter_filepath = './saved_models/selftrain_hybrid1_func2_best_model_for_fold_{}.hdf5'.format(folds)

	for iteration in range(no_iter):
		# x_train, y_train = shuffle(x_train, y_train, random_state=0)

		print("\n Iteration number: {}".format(iteration+1))
		print('Build model...')

		# Input sequences with word indices
		sequences = Input(shape=(maxSeqLength,),dtype='int32', name = 'sequences_input')
		
		# Sequences embedded with word embeddings
		embedding_layer = Embedding(num_words+1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)(sequences)
		embedding_layer_dropout = Dropout(0.25)(embedding_layer)

		# CNN encoder

		# Convolutions
		cnn_layer = Conv1D(filters, 
		                 kernel_size,
		                 padding='valid',
		                 activation='relu',
		                 strides=1)(embedding_layer_dropout)

		# Max pooling  
		cnn_pool_layer = GlobalMaxPooling1D()(cnn_layer)

		# Dense layers and output
		cnn_dense_layer = Dense(hidden_dims)(cnn_pool_layer)
		cnn_dense_dropout = Dropout(0.75)(cnn_dense_layer)
		cnn_output_layer = Activation('relu')(cnn_dense_dropout) 


		# RNN Encoder
		
		lstm_layer = LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences = True)(embedding_layer_dropout)
		attention_layer, word_scores = Attention(return_attention=True)(lstm_layer)

		ling_input_layer = Input(shape=( x_train_ling.shape[1],),dtype='float32', name = 'ling_dense_layer_input')

		ling_dense_layer = Dense(150, activation='relu')(ling_input_layer)
		ling_dropout_layer = Dropout(0.2)(ling_dense_layer)
		ling_dense_layer_2 = Dense(25, activation='relu')(ling_dropout_layer)
		ling_dropout_layer_2 = Dropout(0.2)(ling_dense_layer_2)

		# Concatenate the features extracted from the two encoders and linguistic knowledge. Apply dense layers
		concat_layer = concatenate([cnn_output_layer, attention_layer, ling_dropout_layer_2])
		final_dense_layer = Dense(2)(concat_layer)
		final_layer = Activation('softmax')(final_dense_layer)


		model = Model(inputs=[sequences, ling_input_layer], outputs=final_layer)


		# Use early stopping if the validation loss doesn't decrease in 8 epochs of training. Save the model for that epoch, where the validation loss was 
		earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='min')
		filepath = './saved_models/selftrain_hybrid_latest_ckpt.hdf5'


		class Metrics(keras.callbacks.Callback):
			def on_train_begin(self, logs={}):

				self.f1s = []
				self.recalls = []
				self.precisions = []

			def on_epoch_end(self, epoch, logs={}):
				y_pred = self.model.predict({'sequences_input': self.validation_data[0], 'ling_dense_layer_input': self.validation_data[1]})
				y_pred = np.argmax(y_pred, 1)
				y_valid = np.argmax(self.validation_data[2],1)
				f1_epoch_end = f1_score(y_valid, y_pred)

				self.f1s.append(f1_epoch_end)

		
				print("   Validation F1: {}   ".format(f1_epoch_end))
				logs['val_f1'] = f1_epoch_end
				return

			# def on_train_end(self, epoch, logs={}):	
			# 	precision_kiter.append(self.precisions[-1])
			# 	recall_kiter.append(self.recalls[-1])
			# 	f1_kiter.append(self.f1s[-1])


		metrics = Metrics()

		model.compile(loss='categorical_crossentropy',
		              optimizer='adam',
		              metrics=["accuracy"])
		class_weight = {0 : 1., 1: 10.}

		filepath = './saved_models/selftrain_hybrid1_func2_ckpt.hdf5'
		modelCheckpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only = True, save_weights_only=False, mode='auto', period=1)
		# print(model.summary())


		history = model.fit([x_train, x_train_ling], y_train,
		          batch_size=batch_size,
		          epochs=epochs,
		          validation_data=({'sequences_input': x_val, 'ling_dense_layer_input': x_val_ling}, y_val), 
		          class_weight = class_weight, callbacks=[metrics, earlyStopping, modelCheckpoint], verbose =1, shuffle = True)

		# Testing
		del model
		model = keras.models.load_model(filepath, custom_objects={'Attention': Attention})

		# Finding the precision, recall, F1 for the current model on the test set
		y_pred = model.predict({'sequences_input': x_test, 'ling_dense_layer_input': x_test_ling})
		y_pred = np.argmax(y_pred, axis = 1)
		y_prob = np.asarray(model.predict({'sequences_input': x_test, 'ling_dense_layer_input': x_test_ling}))[:,1]
		y_test_prime = np.argmax(y_test,1)
		accuracy = accuracy_score(y_test_prime, y_pred)
		recall = recall_score(y_test_prime, y_pred)
		precision = precision_score(y_test_prime, y_pred)
		f1 = f1_score(y_test_prime, y_pred)

		f1_kiter.append(f1)
		precision_kiter.append(precision)
		recall_kiter.append(recall)
		print("Iteration number: {}, Fold : {}".format(iteration, folds))
		print("F1: {}".format(f1))
		print("Precision: {}".format(precision))
		print("Recall: {}".format(recall))
		# model.save('./saved_models/hybrid1/hybrid1_{}_{}_{}_without_shuffle.h5'.format(data_genre, iteration, folds))

		# Evaluating the model on the validation data
		y_pred = model.predict({'sequences_input': x_val, 'ling_dense_layer_input': x_val_ling})
		y_pred = np.argmax(y_pred, axis = 1)
		y_prob = np.asarray(model.predict({'sequences_input': x_val, 'ling_dense_layer_input': x_val_ling}))[:,1]
		y_val_prime = np.argmax(y_val,1)
		accuracy = accuracy_score(y_val_prime, y_pred)
		recall = recall_score(y_val_prime, y_pred)
		precision = precision_score(y_val_prime, y_pred)
		f1 = f1_score(y_val_prime, y_pred)

		f1_val_kiter.append(f1)
		precision_val_kiter.append(precision)
		recall_val_kiter.append(recall)

		scores = model.evaluate({'sequences_input': x_val, 'ling_dense_layer_input': x_val_ling}, y_val)		
		mae_val_kiter.append(scores[0])
		acc_val_kiter.append(scores[1])
		# Find the epoch where the best validation loss was recorded
		best_epoch = np.argmin(history.history['val_loss'])
		loss_val_kiter.append(np.min(history.history['val_loss']))
		train_loss_kiter.append(history.history['loss'][best_epoch])


		# Finding most confident predictions on the unlabeled dataset

		# y_pred = model.predict_classes(unlabeled_data)
		y_pred = model.predict({'sequences_input': unlabeled_data, 'ling_dense_layer_input': unlabeled_ling})
		y_pred = np.argmax(y_pred, axis = 1)

		print("Predictions:")
		# print(y_pred)
		positive_class_prediction_indices = [i for i in range(len(y_pred)) if y_pred[i] == 1]

		y_prob = np.asarray(model.predict({'sequences_input': unlabeled_data, 'ling_dense_layer_input': unlabeled_ling}))[:,1]
		sorted_indices = y_prob.argsort()

		most_confident_negative_indices = sorted_indices[:100]

		# print("Most confident positive indices = {}".format(most_confident_positive_indices))
		most_confident_positive_indices = [i for  i in sorted_indices[-100:] if i in positive_class_prediction_indices]

		# print("Most confident negative indices = {}".format(most_confident_negative_indices))

		x_extension_positive = np.asarray([unlabeled_data[i] for i in most_confident_positive_indices])
		xling_extension_positive = np.asarray([unlabeled_ling[i] for i in most_confident_positive_indices])

		x_extension_negative = np.asarray([unlabeled_data[i] for i in most_confident_negative_indices])
		xling_extension_negative = np.asarray([unlabeled_ling[i] for i in most_confident_negative_indices])


		unlabeled_data = np.asarray([unlabeled_data[i] for i in range(len(unlabeled_data)) if(i not in most_confident_positive_indices and i not in most_confident_negative_indices)])
		unlabeled_ling = np.asarray([unlabeled_ling[i] for i in range(len(unlabeled_ling)) if(i not in most_confident_positive_indices and i not in most_confident_negative_indices)])

		print(len(unlabeled_data))

		y_extension_positive = to_categorical(y_pred[most_confident_positive_indices], num_classes = 2)
		y_extension_negative = to_categorical(y_pred[most_confident_negative_indices], num_classes = 2)

		
		print("The dimensions of x_train, x_extension_positive, x_extension_negative = {}, {}, {}".format(x_train.shape, x_extension_positive.shape, x_extension_negative.shape))
		x_train = np.vstack((x_train, x_extension_positive, x_extension_negative))
		y_train = np.vstack((y_train, y_extension_positive, y_extension_negative))
		x_train_ling = np.vstack((x_train_ling, xling_extension_positive, xling_extension_negative))
		K.clear_session()



	f1_kfold.append(f1_kiter)
	recall_kfold.append(recall_kiter)
	precision_kfold.append(precision_kiter)
	train_loss_kfold.append(train_loss_kiter)

	f1_val_kfold.append(f1_val_kiter)
	recall_val_kfold.append(recall_val_kiter)
	precision_val_kfold.append(precision_val_kiter)
	mae_val_kfold.append(mae_val_kiter)
	acc_val_kfold.append(acc_val_kiter)
	loss_val_kfold.append(loss_val_kiter)



print("Final statistics for 5 fold (by taking the mean over all the folds): ")
print("***********")

pickle.dump( recall_kfold, open( "{}_recall_kfold.p".format(data_genre), "wb" ) )	    
pickle.dump( precision_kfold, open( "{}_precision_kfold.p".format(data_genre), "wb" ) )	    
pickle.dump( f1_kfold, open( "{}_f1_kfold.p".format(data_genre), "wb" ) )	   


print("******************************")
print("Evaluation:")

results = (precision_kfold, recall_kfold, f1_kfold)

print("\n\n Stopping the iterations on the basis of the val F1")
print_best_performance(f1_val_kfold, results, 'max')

results_2 = (acc_val_kfold, precision_val_kfold, recall_val_kfold, f1_val_kfold, mae_val_kfold, train_loss_kfold)

results_total = (results, results_2)
# Saving the K fold scores so as to make different plots as and when required
pickle.dump( results_total, open( "{}_results_total_1.p".format(data_genre), "wb" ) )	    

