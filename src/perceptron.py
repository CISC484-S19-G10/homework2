from collections import Counter
import pprint

from main import read, split_data

CLASS_VALUE = 'class_value'
def extract_instances(dir_name, class_value=None):
    raw_instances = read(dir_name, True, combine=list.append)

    def parse_instance(raw_inst):
        inst = Counter(raw_inst)
        inst[CLASS_VALUE] = class_value

        return inst

    #convert trom a raw sequence of strings to a dict containing counts of those strings
    instances = [parse_instance(raw_inst) for raw_inst in raw_instances]
    
    return instances

BIAS = 'bias'
def perceptron_function(instance, weights):
    #always add in the bias
    total = weights[BIAS]

    #sum up each attribute's weighted contribution
    for attr, val in instance.items():
        if CLASS_VALUE != attr:
            total += val * weights[attr]
    
    if total > 0:
        return 1
    else:
        return 0

def train_perceptron(training_data, learning_rate=1/64, initial_weight=lambda x: 1, n_iters=1):
    #find all of the attributes in our training data
    attributes = set()
    for inst in training_data:
        attributes |= set(inst.keys())
    #the class is not an attribute
    attributes.remove(CLASS_VALUE)
    #but the bias kinda is
    attributes.add(BIAS)

    #initalise weights
    weights = {attr: initial_weight(attr) for attr in attributes}

    #pprint.pprint(list(weights.keys()))
    print('instances: {} weights: {} iters: {}'.format(len(training_data), len(weights), n_iters))
    for i in range(n_iters):
        for inst in training_data:
            for key in weights:
                weights[key] += learning_rate * (inst[CLASS_VALUE] \
                                                 - perceptron_function(inst, weights))
        print('done iter {} of {}'.format(i + 1, n_iters))

    #bind the weights to the generic perceptron function
    return lambda inst: perceptron_function(inst, weights)

def get_accuracy(perceptron, testing_data):
    total_correct = 0
    for inst in testing_data:
        if perceptron(inst) == testing_data[CLASS_VALUE]:
            total_correct += 1

    return total_correct / len(testing_data)

def build_perceptron_classifier(class_dirs, class_values):
    #combine the data from each directory of example instances of a class
    data = []
    for class_name, dir_name in class_dirs.items():
        data.extend(extract_instances(dir_name, class_values[class_name]))

    #do a 70:30 split
    SPLIT_PROPS = {'train' : .7, 'valid': .3}
    splits = split_data(data, SPLIT_PROPS)
    training_split, validation_split = splits['train'], splits['valid']
    
    def get_n_iters_accuracy(n_iters):
        perceptron = train_perceptron(training_split, n_iters=n_iters)

        accr = get_accuracy(perceptron, validation_split)

        print('accuracy for {} iters: {}'.format(n_iters, accr))

        return accr

    #find the number of iterations which gives us the best accuracy
    n_iters = max(range(3,6), key=get_n_iters_accuracy)

    return train_perceptron(data, n_iters=n_iters)
