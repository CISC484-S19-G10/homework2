import argparse
import os
import codecs
import re
import math
from collections import Counter

CLASS_VALS = ["spam", "ham"]


def parse_args():
	parser = argparse.ArgumentParser()

	#required args
	parser.add_argument('dir', \
	                    help='the directory containing the datasets')

	#optional args
	parser.add_argument('--train', default='training_set.csv', \
	                    help='the name of the file containing the training data in the given dir')
	parser.add_argument('--test', default='test_set.csv', \
	                    help='the name of the file containg the testing data in the given dir')
	parser.add_argument('--valid', default='validation_set.csv', \
	                    help='the name of the file containg the validation data in the given dir') 

	return parser.parse_args()

def read(read_dir, alpha_only):
    to_return = []
    for filename in os.listdir(read_dir):
        f = codecs.open(os.path.join(read_dir, filename), "r", encoding="us-ascii", errors="ignore")
        lines = f.read()
        if alpha_only:
            lines = re.findall(r"\w+", lines)
        else:
            lines = lines.split()

        to_return.extend(lines)
        f.close()

    return to_return

def read_file(path, alpha_only):
    f = codecs.open(path, "r", encoding="us-ascii", errors="ignore")
    lines = f.read()
    if alpha_only:
        lines = re.findall(r"\w+", lines)
    else:
        lines = lines.split()

    #to_return.extend(lines)
    f.close()

    return lines

def load_files(read_dir):
    texts = []
    for filename in os.listdir(read_dir):
        texts.append(read_file(os.path.join(read_dir,filename),True))
    return texts


def corpus_dict(corpus_dir):
    corpus = read(corpus_dir, True)
    size = len(corpus)
    corpus_dict = Counter(corpus)

    print(size)
    #print(size)
    #for key in corpus_dict:
    #    prob = corpus_dict[key]/size
    #    corpus_dict[key] = math.log(prob)
def corpus_log_prob(corpus_dir, bias=1):
    corpus = read(corpus_dir, True)
    size = len(corpus)
    corpus_dict = Counter(corpus)
    adjusted_size = bias * len(corpus_dict) + size
    #print(size)
    for key in corpus_dict:
		#bias needed so we can compare against tokens which don't appear in the corpus
        prob = (corpus_dict[key] + bias) / adjusted_size
        corpus_dict[key] = math.log(prob)
        #print(prob)
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
        #print(corpus_dict["__DEFAULT__"])
    #total+=document_prob
    #return total
    #print(total)

    #for word in text:
        #print(word+": "+str(corpus_dict[word]))


    #return (len(corpus), Counter(corpus))

def get_naive_bayes_classifier(class_probs, document):
	pass

def count_documents(read_dir):
    countfiles = next(os.walk(read_dir))[2] #dir is your directory path as string
    return len(countfiles)

def accuracy(test_dir, train_probs, counts, class_value):
    total_count = counts["ham"]+counts["spam"]
    doc_probs = {c: counts[c]/total_count for c in CLASS_VALS}
    doc_probs["spam"] = math.log(doc_probs["spam"])
    doc_probs["ham"] = math.log(doc_probs["ham"])

    correct = 0
    total_read = 0
    for item in load_files(test_dir):
        classifications = {c: classify(train_probs[c], item, doc_probs[c]) for c in CLASS_VALS}
        m = max(classifications, key=classifications.get)

        
        if m == class_value:
            correct+=1
        else:
            pass
            #print(classifications)
            #print(item)
        total_read+=1


    return correct/total_read
    #pass

def main():
    args = parse_args()
    args.dir = os.path.abspath(args.dir)

    train_dir = os.path.join(args.dir, "train")
    test_dir = os.path.join(args.dir, "test")

    train_probs = {c: corpus_log_prob(os.path.join(train_dir, c)) for c in CLASS_VALS}
    counts = {c: count_documents(os.path.join(train_dir, c)) for c in CLASS_VALS}
    
    accs = {c: accuracy(os.path.join(test_dir, c), train_probs, counts, c) for c in CLASS_VALS}
    print(accs)



if __name__=='__main__':
	main()
