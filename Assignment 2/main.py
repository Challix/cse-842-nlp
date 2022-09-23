from copy import deepcopy
from os import listdir
import numpy as np


class Trainer:
    def __init__(self):
        self.rate = 0.05
        self.vectors = self.load_vectors()
        self.weight = np.random.choice([0, 1], size=(len(self.words)))

        print(self.vectors[0][:100])

    # load text from file
    def load_doc(self, filename):
        file = open(filename, 'r', encoding='utf-8')
        text = file.read()
        file.close()
        return text
    
    def learn(self):
        # cycle through all review vectors until convergence
        loss = []
        for i in range(len(self.vectors)):
            ref = np.array(self.vectors[i])
            weight = np.array(self.weight)
            output = np.dot(ref, weight)
            # when y is pos and actual is neg
            if output >= 0 and i > 4739:
                self.weight += self.rate * -1
                loss.append(i)
            # when y is neg and actual is pos    
            elif output < 0 and i < 4739:
                self.weight += self.rate * -1
                loss.append(i)

    def load_vectors(self):
        string_lst =  self.load_doc("vectors.txt").replace('[', '').replace(']', '').split('\n')
        return [vector.split(',') for vector in string_lst]

    def weight(self):
        return self.weight
    
    def tensors(self):
        return self.vectors

class Testing:
    def __init__(self, model) -> None:
        self.model = model
    
    def classify(self):
        for i in range(len(self.vectors)):
            ref = np.array(self.vectors[i])
            weight = np.array(self.model.weights())
            output = np.dot(ref, weight)

            # when y is pos and actual is neg
            if output >= 0 and i > 4739:
                self.weight += self.rate * -1
                loss.append(i)
            # when y is neg and actual is pos    
            elif output < 0 and i < 4739:
                self.weight += self.rate * -1
                loss.append(i)


# model
train = Trainer()
test = Testing(train)
