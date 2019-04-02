import math
import main
from collections import Counter
import os
from main import *

CLASS_VALS = ["spam", "ham"]

def load_files(read_dir):
    texts = []
    for filename in os.listdir(read_dir):
        texts.append(read_file(os.path.join(read_dir,filename),True))
    return texts

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

def classify(corpus_dict, text, document_prob):
    #print("Classified")
    #print(text)
    #total = sum(map(lambda word: corpus_dict[word], text))
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

#Get the accuracy of naive bayes on a test directory
def accuracy(test_dir, train_probs, counts, class_value):
    total_count = counts["ham"]+counts["spam"]
    doc_probs = {c: counts[c]/total_count for c in CLASS_VALS}
    doc_probs["spam"] = math.log(doc_probs["spam"])
    doc_probs["ham"] = math.log(doc_probs["ham"])

    correct = 0
    total_read = 0

    for item in load_files(test_dir):
        #Classify the doc with both classifiers
        classifications = {c: classify(train_probs[c], item, doc_probs[c]) for c in CLASS_VALS}
        #Choose the max
        m = max(classifications, key=classifications.get)

        if m == class_value:
            correct+=1
        else:
            pass
        total_read+=1


    return correct/total_read

def naive_bayes_accuracy(train_dir, test_dir):
    train_probs = {c: corpus_log_prob(os.path.join(train_dir, c)) for c in CLASS_VALS}
    counts = {c: count_documents(os.path.join(train_dir, c)) for c in CLASS_VALS}
    accs = {c: accuracy(os.path.join(test_dir, c), train_probs, counts, c) for c in CLASS_VALS}
    print(accs)