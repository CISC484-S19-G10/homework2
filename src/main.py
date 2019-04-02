#!/usr/bin/python3

import argparse
import os
import codecs
import re
import math
import random
from collections import Counter

import perceptron

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
            lines = [line.lower() for line in lines]
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

#split_props contain floats representing the desired relative proportions for each split
EPSILON = .001
def split_data(data, split_props):
    #the total proportion must be 1, (modulo floating point error)
    assert(abs(sum(split_props.values()) - 1) < EPSILON)

    #randomise the order of the data
    random.shuffle(data)

    #split up the data based on the specified proportions
    splits = {}
    offset = 0
    for key, prop in split_props.items():
        split_count = round(prop * len(data))
        splits[key] = data[offset:offset + split_count]
        offset += split_count
    
    #we dshould have assigned all of our data to a split
    assert(len(data) == offset)

    return splits

def main():
    args = parse_args()
    args.dir = os.path.abspath(args.dir)

    CLASS_VALS = {
        "spam": 1, 
        "ham": 0
    }
    train_dir = os.path.join(args.dir, "train")
    class_dirs = {c: os.path.join(train_dir, c) for c in CLASS_VALS}
    test_dir = os.path.join(args.dir, "test")

    train_probs = {c: corpus_log_prob(os.path.join(train_dir, c)) for c in CLASS_VALS}
    test_probs = {c: corpus_log_prob(os.path.join(test_dir, c)) for c in CLASS_VALS}

    percept = perceptron.build_perceptron_classifier(class_dirs, CLASS_VALS)

if __name__=='__main__':
    main()
