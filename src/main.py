#!/usr/bin/python3

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

def read(read_dir, alpha_only, combine=list.extend):
    to_return = []
    for filename in os.listdir(read_dir):
        f = codecs.open(os.path.join(read_dir, filename), "r", encoding="us-ascii", errors="ignore")
        lines = f.read()
        if alpha_only:
            lines = re.findall(r"\w+", lines)
        else:
            lines = lines.split()

        combine(to_return, lines)
        f.close()

    return to_return

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
    corpus_dict['__DEFAULT__'] = bias / adjusted_size

    return corpus_dict

    
    #return (len(corpus), Counter(corpus))

def get_naive_bayes_classifier(class_probs, document):
    pass

def main():
    args = parse_args()
    args.dir = os.path.abspath(args.dir)

    CLASS_VALS = ["spam", "ham"]
    train_dir = os.path.join(args.dir, "train")
    test_dir = os.path.join(args.dir, "test")

    train_probs = {c: corpus_log_prob(os.path.join(train_dir, c)) for c in CLASS_VALS}
    test_probs = {c: corpus_log_prob(os.path.join(test_dir, c)) for c in CLASS_VALS}

    

    #print(spam)
    #print(spam["a"])



if __name__=='__main__':
    main()
