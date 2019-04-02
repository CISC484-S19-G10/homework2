import argparse
import os
import codecs
import re
import math
from collections import Counter
from naive_bayes import *


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





def main():
    args = parse_args()
    args.dir = os.path.abspath(args.dir)

    train_dir = os.path.join(args.dir, "train")
    test_dir = os.path.join(args.dir, "test")

    nb_acc = naive_bayes_accuracy(train_dir, test_dir)
    print(nb_acc)
    



if __name__=='__main__':
	main()
