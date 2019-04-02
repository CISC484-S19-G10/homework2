import argparse
import os
import codecs
import re
import math
from collections import Counter

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

def corpus_dict(corpus_dir):
    corpus = read(corpus_dir, True)
    size = len(corpus)
    corpus_dict = Counter(corpus)

    print(size)
    #print(size)
    #for key in corpus_dict:
    #    prob = corpus_dict[key]/size
    #    corpus_dict[key] = math.log(prob)
        #print(prob)

    return corpus_dict

def classify(corpus_dict, text):
    print("Classified")


    #return (len(corpus), Counter(corpus))

def main():
    args = parse_args()
    args.dir = os.path.abspath(args.dir)

    data_dir = os.path.join(args.dir, "test")
    spam_train = os.path.join(data_dir, "spam")
    ham_train = os.path.join(data_dir, "ham")

    

    #t_dir = os.path.join(args.dir, "train")


    spam_logs = corpus_dict(spam_dir)
    #ham_logs = corpus_dict(ham_dir)

    #print(ham_logs)
    #print(spam)
    #print(spam["a"])



if __name__=='__main__':
	main()