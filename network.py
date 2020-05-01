from keras import models, layers
from genome import Genome
import random


class Network:

    def __init__(self):
        self.model = models.Sequential()
        # self.model.add(layers.Dense(10, activation='relu', input_shape=(224,256,3)))
        # self.model.add(layers.Dense(10, activation='relu'))
        # self.model.add(layers.Dense(12, activation='sigmoid'))
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 32, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(12, activation='softmax'))

    def write_to_genome(self):
        gene = Genome(self.model.get_weights())
        return gene

    def write_to_net(self, genome):
        self.model.set_weights(genome.layers)

    def set_all(self, value):
        arrays = self.model.get_weights()
        array_new = []
        for i in arrays:
            reshaped = i.reshape(-1)
            for n in range(len(reshaped)):
                reshaped[n] = value

            array_new.append(i)
        self.model.set_weights(array_new)

    def random_genome(self):
        arrays = self.model.get_weights()
        array_new = []
        for i in arrays:
            reshaped = i.reshape(-1)
            for n in range(len(reshaped)):
                reshaped[n] = random.uniform(-10, 10)

            array_new.append(i)
        return Genome(array_new)

    def predict(self, inputs):

        value = (self.model.predict(inputs))
        return value[0]
