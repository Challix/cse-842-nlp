# cost = -(prediction*log(actual)) + ((1-prediction)*log(1-actual))
# gradient = (actual - prediction)/m * output

# map each document into a binary vector (len of total vocab)
# weight = weight + Rate * Y [actual 1/-1]
# after train go over test data
# weight is a vector
# weight * test vector = prediction (sign)
# load doc into memory

from collections import deque
from string import punctuation
from copy import deepcopy
from os import listdir
import numpy as np
import math
import re

# imdb review paths
pos_dir = 'archive/test/test/pos'
neg_dir = 'archive/test/test/neg' 

# model word mapping class
class Tensor:
    def __init__(self):
        self.words = self.load_doc("vocab.txt").split('\n')
        # self.vectors = self.load_doc("vectors.txt").split('\n')
        self.tensor = [0 for word in self.words]
        self.master = []
        self.indicies = {}

        # populate indicies dictionary with word -> index
        for i in range(len(self.words)):
            actual = self.words[i].split()
            self.indicies[actual[0]] = i

    # load text from file
    def load_doc(self, filename):
        file = open(filename, 'r', encoding='utf-8')
        text = file.read()
        file.close()
        return text

    # parse and sanitize review words
    def clean_doc(self, doc):
        tokens = doc.replace('<', ' ').replace('>', ' ').split()
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [word for word in tokens if len(word) > 1]
        tokens = [word.lower() for word in tokens]
        return tokens

    # save collection of review vectors to txt file
    def save_list(self, filename):     
        file = open(filename, 'w', encoding='utf-8')
        for vec in range(len(self.master)):
            file.write(str(self.master[vec])+'\n')
        file.close()

    def generate_vectors(self, directory):
        # seperate words in each review into token list
        for filename in listdir(directory):
            path = directory + '/' + filename
            doc = self.load_doc(path)
            tokens = self.clean_doc(doc)

            # build binary vector from token list
            vector = deepcopy(self.tensor)
            for word in tokens:
                if word not in self.indicies.keys():
                    continue
                vector[self.indicies[word]] = 1
            self.master.append(vector)

        print("length of vector", len(self.master[0]))
        print("length of master", len(self.master))

    def update_weight(self, correct):
        v = np.array([2, 1])
        s = np.array([3, -2])
        d = np.dot(v, s)
        change = self.rate * correct
        self.w += change

    # import review vectors from text file
    def load_vectors(self):
        string_lst =  self.load_doc("vectors.txt").replace('[', '').replace(']', '').split('\n')
        return [vector.split(',') for vector in string_lst]
        

class Trainer:
    def __init__(self, tensor):
        self.rate = 0.05
        self.tensor = tensor
        self.vectors = self.tensor.load_vector()
        self.weight = np.random.choice([0, 1], size=(len(self.words)))
    
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


    def gradient_descent(self, gradient, start, n_iter=50, tolerance=1e-06):
        vector = start
        for _ in range(n_iter):
            diff = -self.rate * gradient(vector)
            if np.all(np.abs(diff) <= tolerance):
                break
            vector += diff
        return vector

        
tensor = Tensor()

# create review tensor vectors
# tensor.generate_vectors(pos_dir)
# tensor.generate_vectors(neg_dir)

# trainer = Trainer(tensor)
# print(trainer.gradient_descent(lambda v: 2 * v, 10))
# tensor.save_list("vectors.txt")
