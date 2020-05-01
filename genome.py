import random

import numpy as np


class Genome:
    def __init__(self, layers, mutation=0.05, score=0):
        self.layers = layers
        self.mutation = mutation
        self.score=score

    def cross(self, other):
        child_genes = []
        for i in range(len(self.layers)):  # iterates through layers
            child_genes.append(self.single_cross(self.layers[i], other.layers[i]))
        return Genome(child_genes)

    def single_cross(self, parent1, parent2):
        original_shape = parent1.shape
        parent1_list = parent1.reshape(-1)
        parent2_list = parent2.reshape(-1)
        child = np.zeros(original_shape)
        child = child.reshape(-1)
        for i in range(len(parent1_list)):
            if random.choice([True, False]):
                child[i] = parent1_list[i]
            else:
                child[i] = parent2_list[i]
            if random.random() > 1.0 - self.mutation:
                child[i] = random.uniform(-10, 10)
        return child.reshape(original_shape)

    def __str__(self):
        output = ''
        for n, layer in enumerate(self.layers):
            output += "Layer " + str(n) + ":\n"
            output += str(layer) + "\n"

        return output

    def __lt__(self, other):
        return self.score < other.score
