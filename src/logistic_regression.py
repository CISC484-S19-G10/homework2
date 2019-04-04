import argparse
import os
import codecs
import re
import math
from collections import Counter
import numpy as np
from heapq import nlargest

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

def read(read_dir, alpha_only,combine=list.extend):
    to_return = []
    for filename in os.listdir(read_dir):
        f = codecs.open(os.path.join(read_dir, filename), "r", encoding="us-ascii", errors="ignore")
        lines = f.read()
        if alpha_only:
            lines = re.findall(r"\w+", lines)
        else:
            lines = lines.split()

        combine(to_return,lines)
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

    
def readforTest(read_dir, alpha_only):
    #this one returns back an array of individual emails
    to_return = []
   
    filenames = os.listdir(read_dir)
   
    for filename in filenames:
        f = codecs.open(os.path.join(read_dir, filename), "r", encoding="us-ascii", errors="ignore")
        lines = f.read()
        if alpha_only:
            lines = re.findall(r"\w+", lines)
        else:
            lines = lines.split()

        to_return.append(lines)
        f.close()

    return to_return


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

    # print(corpus_arr)

    return corpus_arr


def logReg(xiwiDict,w0):
    #returns the P[Y=1|Xi]    
    xiwiSum = sum(xiwiDict.values())

    return 1/(1+math.exp(-xiwiSum+w0))

def classify(sample,weights,w0):
    #returns 1 if P[Y=1|Xi] > p[Y=0|Xi] else 0
    weightedSum = 0
    for k in sample:
        try:
            weightedSum += sample[k]*weights[k]
        except:
            weightedSum += .1
    result2 = math.exp(-weightedSum+w0)/(1+math.exp(-weightedSum+w0))
    result = 1/(1+math.exp(-weightedSum+w0))
    # print("SPAM% " + str(result) + " | HAM% " + str(result2))
    if result > result2 :
        return 1
    else:
        return 0


def learnWeights(training_data,learnRate,iterations,startingWeight,lam,q=2,spam=1):    
    #get all attributes from training data
    attributes = set()
    for inst in training_data:
        attributes |= set(inst.keys())

    #create initial weight dictionary
    weights = {attr: startingWeight for attr in attributes}
    
    for i in range(iterations):        
        #expected class value True(1) or False(0)
        classVal = True
        
        weightedAttr = {attr: 0 for attr in attributes}
        for inst in training_data:
            #for each iteration, train the weights on both spam and ham
            for w in weights:
                try:
                    attrVal = inst[w]
                except:
                    attrVal = 0
                weightedAttr[w] = attrVal*weights[w]
            
            classPredict = logReg(weightedAttr,.1)
            err = classVal - classPredict
            
            #flip the value from spam->ham->spam->ham
            classVal = not classVal

            gradient = {attr:inst[attr]*err for attr in attributes}
        
            for k in weights:
                weights[k] += (learnRate*gradient[k])-((lam*0.5)*(weights[k]**2))
    return weights

def get_accuracy(testing_data,classVal,weights):
    total_correct = 0
    for inst in testing_data:
        if classify(inst,weights,.1) == classVal:
            total_correct += 1

    return total_correct / len(testing_data)

def test_logistic_regression(eval_dir,classVal,weights):
    instances = readforTest(eval_dir,True)
    instances_with_counts = []
    for i in instances:
        instances_with_counts.append(Counter(i))
    
    # print("Instances for eval: " + str(len(instances)))

    print(get_accuracy(instances_with_counts,classVal,weights))

def main():
    args = parse_args()
    args.dir = os.path.abspath(args.dir)

    CLASS_VALS = ["spam", "ham"]
    train_dir = os.path.join(args.dir, "train")
    test_dir = os.path.join(args.dir, "test")

    # For Logistic Regression
    train_counts_ham,validate_counts_ham = corpus_counts(os.path.join(train_dir, "ham"),True)
    train_counts_spam,validate_counts_spam = corpus_counts(os.path.join(train_dir, "spam"),True)

    # Combined training set (ham/spam)
    trainHamSpam = [train_counts_spam,train_counts_ham]


    # learn the weights
    weights = learnWeights(trainHamSpam,.0001,3,0,1)

    # check ham results
    print(" HAM %: ",end="")
    test_logistic_regression(os.path.join(test_dir, "ham"),0,weights)    
    
    # check spam results
    print("SPAM %: ",end="")
    test_logistic_regression(os.path.join(test_dir, "spam"),1,weights)


if __name__=='__main__':
	main()
