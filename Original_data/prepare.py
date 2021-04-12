import numpy as np 
import os
from nltk import word_tokenize
import pickle
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk import word_tokenize

strip_special_chars = re.compile("[^A-Za-z0-9\.,;!? ]+")

# 106 was found to be the longest sequence in the dataset
maxSeqLength = 106
embeddings_index = {}
GLOVE_DIR = "../Resources"
EMBEDDING_DIM = 300
embeddings_index = {}


def extractor(path):
	sentences = []
	labels = []
	with open(path, "r") as f:
		for line in f:
			splitted_line = line.strip().split('\t')
			if(len(splitted_line) == 2):
				if(len(splitted_line[0])>0):
					sentences.append(splitted_line[0])
					labels.append(splitted_line[1])
	return(sentences, labels)

def extract():
	genres = ['hotel', 'electronics']
	file_types = ['valid', 'test', 'train']

	for genre in genres:
		for file_type in file_types:
			path = './{}_{}.txt'.format(genre, file_type)
			sentences, labels = extractor(path)
			pickle.dump( sentences, open( './{}_{}_sentences.p'.format(genre, file_type), "wb" ) )
			pickle.dump(labels, open( './{}_{}_labels.p'.format(genre, file_type), "wb" ))

def cleanSentence(string):
	"""
	Converts the sentence into lowercase and removes its special characters
	:param string: the sentence to be cleaned
	:return : cleaned string
	"""
	return re.sub(strip_special_chars, "", string.lower())

def index_word_vectors():
	"""
	Load the Glove word embeddings, convert it into a Numpy array, and save it in a Pickle form.
	"""
	global embeddings_index

	if(os.path.isfile('../extracted/embeddings_index.p')):
		embeddings_index = pickle.load( open( '../extracted/embeddings_index.p', "rb" ) )
		print("Indexed word vectors found.")

	else:	
		print('Indexing word vectors.')

		f = open(os.path.join(GLOVE_DIR, 'glove.42B.300d.txt'), encoding="utf8")
		for line in f:
		    values = line.split()
		    word = values[0]
		    coefs = np.asarray(values[1:], dtype='float32')
		    embeddings_index[word] = coefs
		f.close()
		pickle.dump( embeddings_index, open( "../extracted/embeddings_index.p", "wb" ) )
	print('Found %s word vectors.' % len(embeddings_index))



def generate_embedding_matrix():
	"""
	"""
	genres = ['hotel', 'electronics']
	for genre in genres:
		sentences_train = pickle.load(open('../extracted/{}_train_sentences.p'.format(genre), "rb"), encoding = 'latin1')
		sentences_valid = pickle.load(open('../extracted/{}_valid_sentences.p'.format(genre), "rb"), encoding = 'latin1')
		sentences_test = pickle.load(open('../extracted/{}_test_sentences.p'.format(genre), "rb"), encoding = 'latin1')
		sentences = sentences_train + sentences_valid + sentences_test
		sentences = [cleanSentence(sent_str) for sent_str in sentences]

		labels_train = pickle.load(open('../extracted/{}_train_labels.p'.format(genre), "rb"), encoding = 'latin1')
		labels_valid = pickle.load(open('../extracted/{}_valid_labels.p'.format(genre), "rb"), encoding = 'latin1')
		labels_test = pickle.load(open('../extracted/{}_test_labels.p'.format(genre), "rb"), encoding = 'latin1')
		labels = labels_train + labels_valid + labels_test

		if(genre == 'hotel'):
			sentences_unlabeled = pickle.load(open('./Unlabeled_reviews/corpus-webis-sentences_uppercase.p', "rb"), encoding = 'latin1')
		elif(genre == 'electronics'):
			sentences_unlabeled = pickle.load(open('./Unlabeled_reviews/amazon-50000-sentences_uppercase1.p', "rb"), encoding = 'latin1')
		sentences_unlabeled = [cleanSentence(sent_str) for sent_str in sentences_unlabeled]
	
		sentences_total = sentences + sentences_unlabeled	
		
		
		# finally, vectorize the text samples into a 2D integer tensor
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(sentences_total)

		sequences_labeled = tokenizer.texts_to_sequences(sentences)
		sequences_unlabeled = tokenizer.texts_to_sequences(sentences_unlabeled)

		word_index = tokenizer.word_index
		print('Found %s unique tokens.' % len(word_index))

		sequences_labeled = pad_sequences(sequences_labeled, maxlen=maxSeqLength)
		pickle.dump(sequences_labeled, open("../extracted/{}_data_labeled.p".format(genre), "wb"))

		sequences_unlabeled = pad_sequences(sequences_unlabeled, maxlen=maxSeqLength)
		pickle.dump(sequences_unlabeled, open("../extracted/{}_data_unlabeled.p".format(genre), "wb"))


		print('Preparing embedding matrix.')

		# prepare embedding matrix
		num_words_not_found = 0
		num_words = len(word_index)
		embedding_matrix = np.zeros((num_words + 1, EMBEDDING_DIM))
		for word, i in word_index.items():
			if i >= num_words:
			    continue
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
			    # words not found in embedding index will be all-zeros.
			    embedding_matrix[i] = embedding_vector
			else:
				num_words_not_found+=1

		print("For type {}, word vectors not found = {}".format(genre , num_words_not_found))		
		pickle.dump( embedding_matrix, open( "../extracted/{}_embedding_matrix.p".format(genre), "wb" ) )	    

def extract_tweet_ids():
	for genre in ['hotel']:
		tweet_ids = []
		with open('{}.txt'.format(genre), "r") as f:
			for line in f:
				line = line.strip().split('\t')
				if(len(line) == 3):
					tweet_file_name = line[0]
					tweet_ids.append(tweet_file_name)
		# print(tweet_ids)			
		pickle.dump( tweet_ids, open( "{}_tweet_ids.p".format(genre), "wb" ) )	    
			

def main():
	print("Hi!")
	index_word_vectors()
	generate_embedding_matrix()


if __name__ == '__main__':
	# main()
	extract_tweet_ids()