import math
import main
from collections import Counter
import os
from main import *

CLASS_VALS = ["spam", "ham"]

#Returns an array of texts, which is each an array of words
def load_files(read_dir):
    texts = []
    for filename in os.listdir(read_dir):
        texts.append(read_file(os.path.join(read_dir,filename),True))
    return texts

#Calculates the log probabilities of each word in the corpus
def corpus_log_prob(corpus_dir, bias=1):
    corpus = read(corpus_dir, True)
    size = len(corpus)
    corpus_dict = Counter(corpus)
    adjusted_size = bias * len(corpus_dict) + size
    for key in corpus_dict:
		#bias needed so we can compare against tokens which don't appear in the corpus
        prob = (corpus_dict[key] + bias) / adjusted_size
        corpus_dict[key] = math.log(prob)

	#I'm assuming no one's gonna use this word looking like this, so we should be fine
	#(would be gaureenteed to work if we made everything lowercase in pre-processing...)
    corpus_dict['__DEFAULT__'] = math.log(bias / adjusted_size)

    return corpus_dict

#Add up the log probs to classify the text
def classify(corpus_dict, text, document_prob):
    total = 0
    for word in text:
        if word in corpus_dict:
            total+=corpus_dict[word]
        else:
            total+=corpus_dict["__DEFAULT__"]

    total+=document_prob
    return total


#Count how many documents are in a directory
def count_documents(read_dir):
    countfiles = next(os.walk(read_dir))[2]
    return len(countfiles)

#Get the accuracy of naive bayes model on a test directory
def accuracy(test_dir, train_probs, counts, class_value):
    #Probabilities of each document type
    total_count = counts["ham"]+counts["spam"]
    doc_probs = {c: counts[c]/total_count for c in CLASS_VALS}
    doc_probs["spam"] = math.log(doc_probs["spam"])
    doc_probs["ham"] = math.log(doc_probs["ham"])

    correct = 0
    total_read = 0

    #For each document in the test_directory
    for item in load_files(test_dir):
        #Classify the doc with both classifiers
        classifications = {c: classify(train_probs[c], item, doc_probs[c]) for c in CLASS_VALS}
        #Choose the highest probability
        m = max(classifications, key=classifications.get)

        if m == class_value:
            correct+=1
        else:
            pass
        total_read+=1


    return correct/total_read, correct, total_read

#Get the accuracy on a test sets using the training dir
def naive_bayes_accuracy(train_dir, test_dir):
    #Probability of words in corpus
    train_probs = {c: corpus_log_prob(os.path.join(train_dir, c)) for c in CLASS_VALS}

    #How many of each document we have
    counts = {c: count_documents(os.path.join(train_dir, c)) for c in CLASS_VALS}

    #Running the naive bayes model on the testing data
    accs = {c: accuracy(os.path.join(test_dir, c), train_probs, counts, c) for c in CLASS_VALS}

    #Get results for total accuracy
    total_correct = accs["spam"][1]+accs["ham"][1]
    total_documents = accs["spam"][2]+accs["ham"][2]

    accs["spam"] = accs["spam"][0]
    accs["ham"] = accs["ham"][0]
    accs["total"] = total_correct/total_documents
    return accs
