from collections import Counter
import pprint

from main import read, split_data

CLASS_VALUE = 'class_value'
BIAS = 'bias'

#find all of the attributes in the given data
def get_attributes(data):
    attributes = set()
    for inst in data:
        attributes |= set(inst.keys())
    #the class is not an attribute
    attributes.remove(CLASS_VALUE)
    #but the bias kinda is
    attributes.add(BIAS)

    return attributes

def extract_instances(dir_name, class_value=None, min_attrib_occurences=1):
    raw_instances = read(dir_name, True, combine=list.append)

    def parse_instance(raw_inst):
        inst = Counter(raw_inst)
        inst[CLASS_VALUE] = class_value

        return inst

    #convert trom a raw sequence of strings to a dict containing counts of those strings
    instances = [parse_instance(raw_inst) for raw_inst in raw_instances]

    if min_attrib_occurences > 1:
        #count the number of instances an attribute occurs in
        attrib_counts = {}
        for inst in instances:
            for attrib in inst:
                count, insts = attrib_counts.get(attrib, (0, []))
                insts.append(inst)
                attrib_counts[attrib] = (count + 1, insts)
    
        #get rid of any attributes that don't occur frequently enough
        for attrib in attrib_counts:
            count, insts = attrib_counts[attrib]
            if count < min_attrib_occurences:
                #remove the attribute from any instance that has it
                for insts in insts:
                    insts.pop(attrib)

    return instances

def perceptron_function(instance, weights):
    #always add in the bias
    total = weights[BIAS]

    #sum up each attribute's weighted contribution
    for attr, val in instance.items():
        if CLASS_VALUE != attr:
            total += val * weights.get(attr, 0)
    
    if total > 0:
        return 1
    else:
        return 0

def train_perceptron(training_data, \
                     attributes= None, \
                     learning_rate=1/64, \
                     initial_weight=lambda x: 1, \
                     n_iters=1):
    if None == attributes:
        attributes = get_attributes(training_data)

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
        if perceptron(inst) == inst[CLASS_VALUE]:
            total_correct += 1

    return total_correct / len(testing_data)

def build_perceptron_classifier(class_dirs, class_values):
    #combine the data from each directory of example instances of a class
    data = []
    for class_name, dir_name in class_dirs.items():
        data.extend(extract_instances(dir_name, class_values[class_name], min_attrib_occurences=3))

    #do a 70:30 split
    SPLIT_PROPS = {'train' : .7, 'valid': .3}
    splits = split_data(data, SPLIT_PROPS)
    training_split, validation_split = splits['train'], splits['valid']
    
    #cache the training_split's attributes
    attributes = get_attributes(training_split)

    def get_n_iters_accuracy(n_iters):
        perceptron = train_perceptron(training_split, n_iters=n_iters, attributes=attributes)

        accr = get_accuracy(perceptron, validation_split)

        print('accuracy for {} iters: {}'.format(n_iters, accr))

        return accr

    #find the number of iterations which gives us the best accuracy
    n_iters = max(range(2,6), key=get_n_iters_accuracy)

    return train_perceptron(data, n_iters=n_iters)

def get_accuracy_on_dirs(perceptron, class_dirs, class_values):
    #combine the data from each directory of example instances of a class
    class_data = {class_name: extract_instances(dir_name, class_values[class_name]) \
                     for class_name, dir_name in class_dirs.items()}

    return {c: get_accuracy(perceptron, class_data[c]) for c in class_values}


