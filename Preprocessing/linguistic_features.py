"""
Extract the following features:
1. Presence of common suggestion keywords
2. N gram features of text (unigram, bigram and trigram)
3. N gram features of the POS tags (unigram, bigram and trigram)
4. nsubj dependency relation features
5. Imperative sentence features

Currently run in python 2.7  
Author: Hitesh Golchha
"""
from spacy.symbols import nsubj, PRON

import nltk
import numpy as np 
import spacy
from nltk.corpus import stopwords
import en_core_web_sm

sugg_words_list = ['advice', 'suggest', 'may', 'suggestion', 'ask', 'warn', 'recommend', 'do', 'advise', 'request', 'warning', 'tip', 'recommendation', 'not', 'should', 'can', 'would', 'will']


from nltk.util import ngrams
from nltk.corpus import wordnet as wn
nlp = en_core_web_sm.load()
import pickle



def count_ngram(ngram_list):
	"""
	Given an ngram list return a dictionary having a count of the occurence of the several ngram words, arranged in descending order according to the counts.
	"""
	ngram_count = {}
	for token in ngram_list:
		if token not in ngram_count:
			ngram_count[token] = 1
		else:
			ngram_count[token] += 1
	result = sorted(ngram_count.iteritems(),key=lambda (k,v): (v,k), reverse = True)
	return(result)




def find_ngrams_from_list_of_sentences(sentences):
	text = " ".join(sentences)
	tokenize = nltk.word_tokenize(text)


	unigram = ngrams(tokenize, 1)
	count_unigram = count_ngram(unigram)
	
	bigram = ngrams(tokenize, 2)
	count_bigram = count_ngram(bigram)

	trigram = ngrams(tokenize, 3)
	count_trigram = count_ngram(trigram)



	return (count_unigram, count_bigram, count_trigram)



def find_ngrams_from_pos_sentences(tokenize):

	unigram = ngrams(tokenize, 1)
	count_unigram = count_ngram(unigram)

	bigram = ngrams(tokenize, 2)
	count_bigram = count_ngram(bigram)


	trigram = ngrams(tokenize, 3)
	count_trigram = count_ngram(trigram)


	return (count_unigram, count_bigram, count_trigram)



def imperative_features_extract(file_path):
	"""
	Detecting imperative sentences having clause starting with VB 
	"""
	features = []
	labels = []
	sentences = [] 
	with open(file_path, "r") as train_file:
		line_no = 0
		for line in train_file:
			splitted_line = line.split('\t')
			if(len(splitted_line)!=2):
				continue
			line_no += 1	
			sentence = splitted_line[0]
			sentences.append(sentence)
			label = int(splitted_line[1])
			labels.append(label)
			feature = []
			doc= nlp(unicode(sentence, errors='ignore'))
			found = False
			for token in doc:
				if(token.tag_ == 'VB'):
					vb_subtree = token.subtree
					if(next(vb_subtree).tag_ == 'VB'):
						
						found = True
						break
					else:
						for possible_subj in token.children:
							if(possible_subj.dep == nsubj and possible_subj.text == 'you'):
								
								found = True
								break

			if(found == True):
				features.append([1])
			else:
				features.append([0])
	
	features = np.asarray(features)
	return(features)




def ngram_features_extract(file_path, requisites):
	uni_keys, bi_keys, tri_keys = requisites

	features = []

	with open(file_path, "r") as train_file:
		line_no = 0
		for line in train_file:
			splitted_line = line.split('\t')
			if(len(splitted_line)!=2):
				continue
			line_no += 1	
			sentence = splitted_line[0]
			feature = []
			tokenized_sent = nltk.word_tokenize(sentence)
			sent_unigram = ngrams(tokenized_sent, 1)
			sent_bigram = ngrams(tokenized_sent, 2)
			sent_trigram = ngrams(tokenized_sent, 3)

			for key in uni_keys:
				if(key in sent_unigram):
					feature.append(1)
				else:
					feature.append(0)

			for key in bi_keys:
				if(key in sent_bigram):
					feature.append(1)
				else:
					feature.append(0)	

			for key in tri_keys:
				if(key in sent_trigram):
					feature.append(1)
				else:
					feature.append(0)		
			features.append(feature)
	
	features = np.asarray(features)
	return features					


def dep_features_extract(file_path, requisites):
	nsubj_dependency_relations = requisites

	with open(file_path, "r") as train_file:
		features = []
		line_no = 0
		for line in train_file:
			splitted_line = line.split('\t')
			if(len(splitted_line)!=2):
				continue
			line_no += 1	
			sentence = splitted_line[0]
			feature = np.zeros(len(nsubj_dependency_relations))
			
			doc= nlp(unicode(sentence, errors='ignore'))
			for token in doc:
				if(token.dep == nsubj):
					 possible_feature = ( token.tag_ , token.head.tag_) 
					 if(possible_feature in nsubj_dependency_relations.keys()):
					 	feature[nsubj_dependency_relations[possible_feature]] = 1
			features.append(feature)		 	
	
	features = np.asarray(features)
	return(features) 			



def extract_pos_tag_features(file_path, requisites):
	uni_keys, bi_keys, tri_keys = requisites

	features = []

	with open(file_path, "r") as train_file:
		line_no = 0
		for line in train_file:
			splitted_line = line.split('\t')
			if(len(splitted_line)!=2):
				continue
			line_no += 1	
			sentence = splitted_line[0]
			feature = []

			doc = nlp(unicode(sentence, errors='ignore'))
			tokenized_pos_sent = [token.tag_ for token in doc]

			sent_unigram = ngrams(tokenized_pos_sent, 1)
			sent_bigram = ngrams(tokenized_pos_sent, 2)
			sent_trigram = ngrams(tokenized_pos_sent, 3)

			for key in uni_keys:
				if(key in sent_unigram):
					feature.append(1)
				else:
					feature.append(0)

			for key in bi_keys:
				if(key in sent_bigram):
					feature.append(1)
				else:
					feature.append(0)	

			for key in tri_keys:
				if(key in sent_trigram):
					feature.append(1)
				else:
					feature.append(0)		
			features.append(feature)
	
	features = np.asarray(features)
	return features					


def keyword_features_extract(file_path):
	features = []

	with open(file_path, "r") as train_file:
		line_no = 0
		for line in train_file:
			splitted_line = line.split('\t')
			if(len(splitted_line)!=2):
				continue
			line_no += 1	
			sentence = splitted_line[0]
			feature = []

			doc = nlp(unicode(sentence, errors='ignore'))
			feature = []
			for sugg_keyword in sugg_words_list:
				found = False
				for token in doc:
					if(token.lemma_ == sugg_keyword):
						found = True
						break
				if(found == True):
					feature.append(1)
				else:
					feature.append(0)
			features.append(feature)					

	features = np.asarray(features)
	return features					


def extract_labels(file_path):
	labels = []
	with open(file_path, "r") as train_file:
		line_no = 0
		for line in train_file:
			splitted_line = line.split('\t')
			if(len(splitted_line)!=2):
				continue
			line_no += 1
			labels.append(int(splitted_line[1]))
	labels = np.asarray(labels)
	return labels			




def extract_features(file_path, requisites):
	print("Extracting features for file : {}".format(file_path))

	ngram_requisites, pos_ngram_requisites, dep_features_requisites = requisites


	ngram_features = ngram_features_extract(file_path, ngram_requisites)
	dep_features = dep_features_extract(file_path, dep_features_requisites)
	pos_ngram_features = extract_pos_tag_features(file_path, pos_ngram_requisites)
	imperative_features = imperative_features_extract(file_path)
	keyword_features = keyword_features_extract(file_path)
	
	# print("Printing shapes:")
	# print(ngram_features.shape)
	# print(dep_features.shape)
	# print(pos_ngram_features.shape)
	# print(imperative_features.shape)
	# print(keyword_features.shape)


	features = np.concatenate((ngram_features, dep_features, pos_ngram_features, imperative_features, keyword_features), axis = 1)
	labels = extract_labels(file_path)
	result = (features, labels)
	return result


def prepare_features(file_type):
	"""
	Firstly the training file is used to identify what will be the features to be extracted, on the basis of which the classification shall be done. 
	The prerequisites for those specific feature extraction is performed in this file.
	
	: param file_type: genre of dataset 
	"""

	print("Preparing features for file_type: {}".format(file_type))
	ngram_requisites = prepare_ngram_features(file_type)
	pos_ngram_requisites = prepare_pos_ngram_features(file_type)
	dep_features_requisites = prepare_dep_features(file_type)
	result = (ngram_requisites, pos_ngram_requisites, dep_features_requisites)
	return result


def first_n_pairs_from_dict(mydict, n):
	first_n_pairs = {k: mydict[k] for k in mydict.keys()[:n]}
	return(first_n_pairs)


def prepare_ngram_features(file_type, unigram_number = 300, bigram_number = 100, trigram_number = 100):
	"""


	"""
	# Currently preparing ngram features for all sentences. Specifically the ngram values for only suggestive sentences may be part of a future experiment.


	sentences = []
	suggestive_sentences = []
	non_suggestive_sentences = []

	with open('../Data/Processed/{}_train.txt'.format(file_type), "r") as train_file:
		line_no = 0
		for line in train_file:
			splitted_line = line.split('\t')
			if(len(splitted_line)!=2):
				continue
			line_no += 1	
			sentence = splitted_line[0]
			label = int(splitted_line[1])
			sentences.append(sentence)
			if(label == 1):
				suggestive_sentences.append(sentence)
			else:
				non_suggestive_sentences.append(sentence)
				
		
	uni, bi, tri = find_ngrams_from_list_of_sentences(sentences)
	uni = uni[:unigram_number]
	bi = bi[:bigram_number]
	tri = tri[:trigram_number]

	uni = [key for (key, value) in uni]
	bi =  [key for (key, value) in bi]
	tri =  [key for (key, value) in tri]

	# uni_10 = uni[:10]
	# print("Unigram: {}".format(uni_10))

	# bi_10 = bi[:10]
	# print("Biigram: {}".format(bi_10))

	# tri_10 = tri[:10]
	# print("Trigram: {}".format(tri_10))



	result = (uni, bi, tri)

	return result 

def prepare_pos_ngram_features(file_type, unigram_number = 50, bigram_number = 50, trigram_number = 50):
	"""
	"""
	pos_sentences = []
	pos_sugg_sentences = []
	pos_non_sugg_sentences = []

	with open('../Data/Processed/{}_train.txt'.format(file_type), "r") as train_file:
		line_no = 0
		for line in train_file:
			splitted_line = line.split('\t')
			if(len(splitted_line)!=2):
				continue
			line_no += 1	
			sentence = splitted_line[0]
			label = int(splitted_line[1])
			doc= nlp(unicode(sentence, errors='ignore'))
			pos_sent = [token.tag_ for token in doc]
			pos_sentences = pos_sentences + pos_sent
			if(label == 1):
				pos_sugg_sentences+= pos_sent
			else:
				pos_non_sugg_sentences+=pos_sent
	
	# print("\n N grams of all sentences:")			
	uni, bi, tri = find_ngrams_from_pos_sentences(pos_sentences)

	uni = uni[:unigram_number]
	bi = bi[:bigram_number]
	tri = tri[:trigram_number]

	uni = [key for (key, value) in uni]
	bi =  [key for (key, value) in bi]
	tri =  [key for (key, value) in tri]

	# uni_10 = uni[:10]
	# print("Unigram: {}".format(uni_10))
	# bi_10 = bi[:10]
	# print("Biigram: {}".format(bi_10))

	# tri_10 = tri[:10]

	# print("Trigram: {}".format(tri_10))

	

	result =  (uni, bi, tri) 

	return result 

def prepare_dep_features(file_type):
	"""
	Extract the Dependency relation features for the dataset.
	: param file_type: The genre of dataset ('electronics/hotel')
	"""
	nsubj_dependency_relations = {}

	with open('../Data/Processed/{}_train.txt'.format(file_type), "r") as train_file:
		features = []
		line_no = 0
		for line in train_file:
			splitted_line = line.split('\t')
			if(len(splitted_line)!=2):
				continue
			line_no += 1	
			sentence = splitted_line[0]
			label = int(splitted_line[1])
			feature = []
			doc= nlp(unicode(sentence, errors='ignore'))
			for token in doc:
				if(token.dep == nsubj):
					feature.append( ( token.tag_ , token.head.tag_) )
			features.append(feature)
	for feature in features:
		for relation in feature:
			if(relation not in nsubj_dependency_relations.keys()):
				nsubj_dependency_relations[relation] = len(nsubj_dependency_relations)

	return nsubj_dependency_relations


	
def main():

	# Unlabeled sentences text file : ../corpus-webis-sentences_uppercase.p amazon-50000-sentences_uppercase1.p




	# For Hotel
	requisites = prepare_features('hotel')
	hotel_train_features, hotel_train_labels = extract_features('../Data/Processed/hotel_train.txt', requisites)
	pickle.dump(hotel_train_features, open("../Data/Linguistic_features/{}/hotel_train_features.p".format('hotel'), "wb"))
	pickle.dump(hotel_train_labels, open("../Data/Linguistic_features/{}/hotel_train_labels.p".format('hotel'), "wb"))



	hotel_test_features, hotel_test_labels = extract_features('../Data/Processed/hotel_test.txt', requisites)
	pickle.dump(hotel_test_features, open("../Data/Linguistic_features/{}/hotel_test_features.p".format('hotel'), "wb"))
	pickle.dump(hotel_test_labels, open("../Data/Linguistic_features/{}/hotel_test_labels.p".format('hotel'), "wb"))


	hotel_valid_features, hotel_valid_labels = extract_features('../Data/Processed/hotel_valid.txt', requisites)
	pickle.dump(hotel_valid_features, open("../Data/Linguistic_features/{}/hotel_valid_features.p".format('hotel'), "wb"))
	pickle.dump(hotel_valid_labels, open("../Data/Linguistic_features/{}/hotel_valid_labels.p".format('hotel'), "wb"))

	hotel_unlabeled_features, _ = extract_features('../Original_data/Unlabeled_reviews/Text/corpus-webis-sentences_uppercase.txt', requisites)
	pickle.dump(hotel_unlabeled_features, open("../Data/Linguistic_features/{}/hotel_unlabeled_features.p".format('hotel'), "wb"))

	# For Electronics
	requisites = prepare_features('electronics')
	electronics_train_features, electronics_train_labels = extract_features('../Data/Processed/electronics_train.txt', requisites)
	pickle.dump(electronics_train_features, open("../Data/Linguistic_features/{}/electronics_train_features.p".format('electronics'), "wb"))
	pickle.dump(electronics_train_labels, open("../Data/Linguistic_features/{}/electronics_train_labels.p".format('electronics'), "wb"))

	electronics_test_features, electronics_test_labels = extract_features('../Data/Processed/electronics_test.txt', requisites)
	pickle.dump(electronics_test_features, open("../Data/Linguistic_features/{}/electronics_test_features.p".format('electronics'), "wb"))
	pickle.dump(electronics_test_labels, open("../Data/Linguistic_features/{}/electronics_test_labels.p".format('electronics'), "wb"))


	electronics_valid_features, electronics_valid_labels = extract_features('../Data/Processed/electronics_valid.txt', requisites)
	pickle.dump(electronics_valid_features, open("../Data/Linguistic_features/{}/electronics_valid_features.p".format('electronics'), "wb"))
	pickle.dump(electronics_valid_labels, open("../Data/Linguistic_features/{}/electronics_valid_labels.p".format('electronics'), "wb"))	

	electronics_unlabeled_features, _ = extract_features('../Original_data/Unlabeled_reviews/Text/amazon-50000-sentences_uppercase1.txt', requisites)
	pickle.dump(electronics_unlabeled_features, open("../Data/Linguistic_features/{}/electronics_unlabeled_features.p".format('electronics'), "wb"))
	# pickle.dump(electronics_unlabeled_labels, open("../Data/Linguistic_features/{}/electronics_unlabeled_labels.p".format('electronics'), "wb"))	



	# Consistency checks

	data_genres = ['electronics', 'hotel']
	data_types = ['train', 'test', 'valid']

		# Consistency check
	for data_genre in data_genres:
		for data_type in data_types:
			# x = pickle.load(open('../Data/Processed/{}/data_{}.p'.format(data_genre, data_type), "rb"))
			# no_of_sentences = len(x)
			no_of_sentences2 = len(eval("{}_{}_labels".format(data_genre, data_type)))
			print("{}-{} length = {}".format(data_genre, data_type, no_of_sentences2))






	



if __name__ == '__main__':
	main()