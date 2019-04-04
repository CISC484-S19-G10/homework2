#!/usr/bin/python3

import argparse
import os
import codecs
import re
import math
import random
from collections import Counter
from naive_bayes import *

from perceptron import get_accuracy_on_dirs, build_perceptron_classifier

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
    
    #if we've missed any instances, assign the remaining ones
    #to the split with the highest rounding error
    while len(data) > offset:
        key = max(split_props.keys(), key=lambda key: split_props[key] - len(splits[key]))
        splits[key].append(data[offset])
        offset += 1

    #we should have assigned all of our data to a split
    assert(len(data) == sum([len(splits[key]) for key in splits]))

    return splits

def main():
    args = parse_args()
    args.dir = os.path.abspath(args.dir)

    CLASS_VALS = {
        "spam": 1, 
        "ham": 0
    }
    train_dir = os.path.join(args.dir, "train")
    test_dir = os.path.join(args.dir, "test")
    
    train_class_dirs = {c: os.path.join(train_dir, c) for c in CLASS_VALS}
    test_class_dirs = {c: os.path.join(test_dir, c) for c in CLASS_VALS}

    nb_acc = naive_bayes_accuracy(train_dir, test_dir)

    print("Accuracy of Naive Bayes Classifier:")
    print(nb_acc)
    
    perceptron = build_perceptron_classifier(train_class_dirs, CLASS_VALS)
    percept_accr = get_accuracy_on_dirs(perceptron, test_class_dirs, CLASS_VALS)

    print("Accuracy of Perceptron Classifier:")
    print(percept_accr)

if __name__=='__main__':
    main()
