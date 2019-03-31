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

def readWithValidateSplit(read_dir, alpha_only):
    to_return = []
    to_return_train = []
    to_return_validate = []

    filenames = os.listdir(read_dir)
    validationStartIndex = math.floor(.7*len(filenames))

    for filename in filenames[:validationStartIndex]:
        f = codecs.open(os.path.join(read_dir, filename), "r", encoding="us-ascii", errors="ignore")
        lines = f.read()
        if alpha_only:
            lines = re.findall(r"\w+", lines)
        else:
            lines = lines.split()

        to_return_train.extend(lines)
        f.close()

    for filename in filenames[validationStartIndex:]:
        f = codecs.open(os.path.join(read_dir, filename), "r", encoding="us-ascii", errors="ignore")
        lines = f.read()
        if alpha_only:
            lines = re.findall(r"\w+", lines)
        else:
            lines = lines.split()

        to_return_validate.extend(lines)
        f.close()

    return to_return_train,to_return_validate

    
def corpus_counts(corpus_dir,trainValSplit):
    corpus_arr = []

    if trainValSplit:
        corpus_train,corpus_validate = readWithValidateSplit(corpus_dir,True)
        corpus_train_dict = Counter(corpus_train)
        corpus_validate_dict = Counter(corpus_validate)
        corpus_arr.append(corpus_train_dict)
        corpus_arr.append(corpus_validate_dict)
    else:
        corpus = read(corpus_dir, True)
        corpus_dict = Counter(corpus)
        corpus_arr.append(corpus_dict)

    print(corpus_arr)

    return corpus_arr

def main():
    args = parse_args()
    args.dir = os.path.abspath(args.dir)

    CLASS_VALS = ["spam", "ham"]
    train_dir = os.path.join(args.dir, "train")
    test_dir = os.path.join(args.dir, "test")

    # For Logistic Regression
    train_counts_ham,validate_counts_ham = corpus_counts(os.path.join(train_dir, "ham"),True)
    train_counts_spam,validate_counts_spam = corpus_counts(os.path.join(train_dir, "spam"),True)

    #print(spam)
    #print(spam["a"])


if __name__=='__main__':
	main()
