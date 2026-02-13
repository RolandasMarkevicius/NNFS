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
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.bias

    def backward(self, gradients):
        #update the weights
        self.weights += 0.01 * np.dot(self.inputs.T, gradients)

        #update the bias
        self.bias += 0.01 * np.sum(gradients)

        #return outputs for further backpropagation
        self.gradients = np.dot(gradients, self.weights.T)
   
class Activation:
    def forward(self, input):
        self.output = np.maximum(0, input)

    def backward(self, gradients):
        self.gradients = 1 if gradients >= 0 else 0

class Softmax:
    def forward(self, input):
        negative_exponents = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = negative_exponents / np.sum(negative_exponents, axis=1, keepdims=True)

class Loss:
    def calculate(self, input, labels):
        loss, accuracy = self.forward(input, labels)
        avg_loss = np.mean(loss)

        avg_accuracy = np.mean(accuracy)

        return avg_loss, avg_accuracy
    
        
class CCE_loss(Loss):
    def forward(self, input, labels):
        np_labels = np.array(labels)
        input_clipped = np.clip(input, 1e-7, 1-1e-7)
        largest_index = np.argmax(input_clipped, axis=1)

        if len(np_labels.shape) == 1:
            yhat = input_clipped[range(len(input_clipped)), np_labels]
            accuracy = largest_index == np_labels

        elif len(np_labels.shape) == 2:
            yhat = np.sum((input_clipped * np_labels), axis=1) #what does keep dims do?
            accuracy = largest_index == np.argmax(np_labels, axis=1)

        else:
            print('this did not work')

        loss = -np.log(yhat)

        return loss, accuracy

#define the dataset
x, y = spiral_data(samples=100, classes=3)

#network architecture
l1 = Layer(2,3)
l1_relu = Activation()
l2 = Layer(3,3)
l2_softmax = Softmax()
loss_function = CCE_loss()

#forward pass
l1.forward(x)
l1_relu.forward(l1.output)

l2.forward(l1_relu.output)
l2_softmax.forward(l2.output)

#loss
loss = loss_function.calculate(l2_softmax.output, y)
print(loss)

#outputs
# print(loss.output)