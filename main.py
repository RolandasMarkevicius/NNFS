import numpy as np
import matplotlib as plt
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer:
    def __init__(self, nr_inputs, nr_neurons):
        self.weights = 0.01 * np.random.randn(nr_inputs, nr_neurons)
        self.bias = np.zeros((1, nr_neurons)) 

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias

class Activation:
    def forward(self, input):
        self.output = np.maximum(0, input)

class Softmax:
    def forward(self, input):
        negative_exponents = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = negative_exponents / np.sum(negative_exponents, axis=1, keepdims=True)

#define the dataset
x, y = spiral_data(samples=100, classes=3)

#network architecture
l1 = Layer(2,3)
l1_relu = Activation()
l2 = Layer(3,3)
l2_softmax = Softmax()

#forward pass
l1.forward(x)
l1_relu.forward(l1.output)

l2.forward(l1_relu.output)
l2_softmax.forward(l2.output)

print(l2_softmax.output[:5])