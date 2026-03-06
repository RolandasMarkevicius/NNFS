import numpy as np
import matplotlib as plt
import nnfs
import os
import urllib
import urllib.request

from nnfs.datasets import spiral_data, sine_data
from zipfile import ZipFile

nnfs.init()

class Layer():
    def __init__(self, nr_inputs, nr_neurons, l1_lambda_weights=0, l1_lambda_bias=0, l2_lambda_weights=0, l2_lambda_bias=0):
        self.weights = 0.01 * np.random.randn(nr_inputs, nr_neurons) #does this implementation need to change?
        self.bias = np.zeros((1, nr_neurons))

        self.l1_lambda_weights = l1_lambda_weights
        self.l2_lambda_weights = l2_lambda_weights
        self.l1_lambda_bias = l1_lambda_bias
        self.l2_lambda_bias = l2_lambda_bias

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.bias

    def backward(self, gradients):
        #update the weights
        self.d_weights = np.dot(self.inputs.T, gradients)

        #update the weights with regularization
        if self.l1_lambda_weights > 0:
            d_l1_weights = np.ones_like(self.d_weights)
            d_l1_weights[self.weights <= 0] = -1
            self.d_weights += self.l1_lambda_weights * d_l1_weights #why are we summing this?

        #update the weights with l2 regualrization
        if self.l2_lambda_weights > 0:
            d_l2_weights = self.l2_lambda_weights * 2 * self.weights
            self.d_weights += d_l2_weights

        #update the bias
        self.d_bias = np.sum(gradients, axis=0, keepdims=True)

        #update the bias with regularization
        if self.l1_lambda_bias > 0:
            d_l1_bias = np.ones_like(self.d_bias)
            d_l1_bias[self.bias <= 0] = -1
            self.d_bias += self.l1_lambda_bias * d_l1_bias

        if self.l2_lambda_bias > 0:
            d_l2_bias = self.l2_lambda_bias * 2 * self.bias
            self.d_bias += d_l2_bias

        #return outputs for further backpropagation
        self.gradients = np.dot(gradients, self.weights.T)

class Init_layer():
    def forward(self, input):
        self.output = input

class Dropout():
    def __init__(self, keep_rate):
        self.keep_rate = keep_rate
    
    def forward(self, inputs):
        self.binary_mask = np.random.binomial(1, self.keep_rate, np.shape(inputs))
        self.output = inputs * self.binary_mask / self.keep_rate

    def backward(self, gradients):
        self.gradients = gradients * self.binary_mask / self.keep_rate

class Linear():
    def forward(self, input):
        self.output = input

    def backward(self, gradients):
        self.gradients = gradients.copy() #is a copy detrimental here?

    def prediction(self, outputs):
        return outputs

class ReLU():
    def forward(self, input):
        self.inputs = input
        self.output = np.maximum(0, input)

    def backward(self, gradients):
        self.gradients = gradients.copy()
        self.gradients[self.inputs <= 0] = 0 

    def prediction(self, output):
        return output

class Softmax():
    def forward(self, input):
        negative_exponents = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = negative_exponents / np.sum(negative_exponents, axis=1, keepdims=True)
    
    def prediction(self, output):
        return np.argmax(output, axis=1)

class Sigmoid():
    def forward(self, input):
        self.output = 1 / (1 + np.exp(-input))

    def backward(self, gradients):
        self.gradients = gradients * self.output * (1 - self.output)

    def prediction(self, output):
        return (output > 0.5) * 1
    
class Accuracy():
    def calculate(self, input, label):
        sample_accuracy = self.compare(input, label)
        batch_accuracy = np.mean(sample_accuracy)
        return batch_accuracy
    
class Regression_accuracy(Accuracy):
    def __init__(self):
        self.acc_target = None

    def init(self, label, reinit=False):
        if self.acc_target == None or reinit:
            self.acc_target = np.std(label) / 250

    def compare(self, input, label):
        sample_accuracy = np.mean(np.abs(input - label) < self.acc_target)

        return sample_accuracy
    
class CCE_accuracy(Accuracy):
    def __init__(self, *, label_one_hot=False):
        #define a switch for labels to be either one hot vectors or argmax integers
        self.label_one_hot = label_one_hot

    def init(self, label, reinit=False):
        #pass the function as no initialisation is required
        pass

    def compare(self, input, label):
        #If the labels are one hot change them to a list
        if self.label_one_hot == True or len(label.shape) == 2:
            self.label = np.argmax(label, axis=1)

        else:
            self.label = label

        #compare inputs with labels to get a True or False output
        comparison = input == self.label
        return comparison

class BCE_accuracy(Accuracy):
    def __init__(self):
        pass

    def init(self, label, reinit=False):
        pass

    def compare(self, input, label):
        comparison = input == label
        
        return comparison

class Loss():
    def regularization_loss(self, weight_layer_list):
        #l1_loss = sum of absolute value of weights/bias times lambda
    
        regularization_loss = 0

        for layer in weight_layer_list:

            #l1_weights
            if layer.l1_lambda_weights > 0:
                regularization_loss += layer.l1_lambda_weights * np.sum(np.absolute(layer.weights))

            #l1_bias
            if layer.l1_lambda_bias > 0:
                regularization_loss += layer.l1_lambda_bias * np.sum(np.absolute(layer.bias))
            
            #l2_weights
            if layer.l2_lambda_weights > 0:
                regularization_loss += layer.l2_lambda_weights * np.sum(layer.weights ** 2)

            #l2_bias
            if layer.l2_lambda_bias > 0:
                regularization_loss += layer.l2_lambda_bias * np.sum(layer.bias ** 2)

        return regularization_loss

    def calculate(self, input, labels, weight_layer_list):
        forward_loss = self.forward(input, labels)
        reg_loss = self.regularization_loss(weight_layer_list=weight_layer_list)

        avg_loss = np.mean(forward_loss)
        #total_loss = avg_loss + reg_loss

        return avg_loss, reg_loss

class MSE_loss(Loss):
    def forward(self, input, label):
        self.sqr_diff = (label - input) ** 2
        loss = np.mean(self.sqr_diff, axis=-1)

        return loss

    def backward(self, input, label):
        self.sample_count = np.array(input).shape[0]
        self.gradients = -2 / self.sample_count * (label - input)

class MAE_loss(Loss):
    def forward(self, input, label):
        self.loss = np.mean(np.abs(input - label), axis=-1)
        self.acc_value = np.std(label) / 250
        self.accuracy = np.mean(np.abs(input - label) < self.acc_value)

        return self.loss, self.accuracy

    def backward(self, input, label):
        self.sample_count = input.shape[0]
        self.feature_count = input.shape[1]
        self.d_loss_per_feature = np.sign(input - label) / self.feature_count
        self.gradients = self.d_loss_per_feature / self.sample_count

class CCE_loss(Loss):
    def forward(self, input, labels):
        np_labels = np.array(labels)
        input_clipped = np.clip(input, 1e-7, 1-1e-7)

        if len(np_labels.shape) == 1:
            yhat = input_clipped[range(len(input_clipped)), np_labels]

        elif len(np_labels.shape) == 2:
            yhat = np.sum((input_clipped * np_labels), axis=1)

        else:
            print('this did not work')

        loss = -np.log(yhat)

        return loss

class Combined_Softamx_and_CCE_loss():
    def backward(self, softmax_outputs, labels):
        #define sample count
        self.sample_count = softmax_outputs.shape[0]

        #adjust the shape of the true values to be not 1hot encoded
        if len(labels.shape) == 2:
            labels = np.argmax(labels, axis=1)

        #if statement to manage true value shape
        self.gradients = softmax_outputs.copy()
        self.gradients[range(self.sample_count), labels] -= 1 #true values in this case is 1 so subtracting one for each sample in the batch

        #noramlise gradients
        self.gradients = self.gradients / self.sample_count

class BCE_loss(Loss):
    def forward(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        self.input_clipped = np.clip(self.inputs, 1e-7, 1-1e-7)

        self.loss_per_sample = self.labels * -np.log(self.input_clipped) + (1 - self.labels) * -np.log(1 - self.input_clipped)
        self.loss = np.mean(self.loss_per_sample, axis=-1)

        return self.loss

    def backward(self, gradients, labels):
        #clip incoming gradients
        clipped_gradients = np.clip(gradients, 1e-7, 1-1e-7)
        nr_of_samples = clipped_gradients.shape[0]

        #calculate the gradients
        self.gradients = - (1 / nr_of_samples) * (labels / clipped_gradients - (1 - labels) / (1 - clipped_gradients))

class Optimizer_SGD_with_momentum():
    def __init__(self, learning_rate=1, decay=0.01, memory=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.memory = memory
        self.itterations = 0

    def pre_optimize(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + (self.decay * self.itterations))

    def optimize(self, layer):
        if self.memory:
            #conditional for initialization (0 array for updated weights)
            if not hasattr(layer, 'weight_memory'):
                #create a new empty array of 0's in the shape of the weights and biases
                layer.weight_memory = np.zeros_like(layer.d_weights)
                layer.bias_memory = np.zeros_like(layer.d_bias)

            layer.weight_memory = (layer.weight_memory * self.memory) + ((1 - self.memory) * layer.d_weights)
            self.weight_updates = layer.weight_memory
            layer.bias_memory = (layer.bias_memory * self.memory) + ((1 - self.memory) * layer.d_bias)
            self.bias_updates = layer.bias_memory
            
            layer.weights -= self.current_learning_rate * self.weight_updates
            layer.bias -= self.current_learning_rate * self.bias_updates

        else:
            layer.weights -= self.current_learning_rate * layer.d_weights
            layer.bias -= self.current_learning_rate * layer.d_bias

    def post_optimize(self):
        self.itterations += 1

class Optimizer_Adagrad():
    def __init__(self, learning_rate=1, decay=0.01, eps=0.0000001):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.itterations = 0
        self.eps = eps

    def pre_optimize(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + (self.decay * self.itterations))

    def optimize(self, layer):
        #conditional for initialization (0 array for updated weights)
        if not hasattr(layer, 'weight_cache'):
            #create a new empty array of 0's in the shape of the weights and biases
            layer.weight_cache = np.zeros_like(layer.d_weights)
            layer.bias_cache = np.zeros_like(layer.d_bias)

        layer.weight_cache += layer.d_weights ** 2
        layer.bias_cache += layer.d_bias ** 2

        layer.weights -= self.current_learning_rate / (np.sqrt(layer.weight_cache) + self.eps) * layer.d_weights
        layer.bias -= self.current_learning_rate / (np.sqrt(layer.bias_cache) + self.eps) * layer.d_bias

    def post_optimize(self):
        self.itterations += 1

class Optimizer_RMSProp():
    def __init__(self, learning_rate=1, decay=0.01, eps=0.0000001, beta=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.eps = eps
        self.beta = beta

    def pre_optimize(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + (self.decay * self.iterations))

    def optimize(self, layer):
        #conditional for initialization (0 array for updated weights)
        if not hasattr(layer, 'weight_cache'):
            #create a new empty array of 0's in the shape of the weights and biases
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.bias)

        layer.weight_cache = layer.weight_cache * self.beta + ((1 - self.beta) * (layer.d_weights ** 2))
        layer.bias_cache = layer.bias_cache * self.beta + ((1- self.beta) * (layer.d_bias ** 2))

        layer.weights -= self.current_learning_rate / (np.sqrt(layer.weight_cache) + self.eps) * layer.d_weights
        layer.bias -= self.current_learning_rate / (np.sqrt(layer.bias_cache) + self.eps) * layer.d_bias

    def post_optimize(self):
        self.iterations += 1

class Optimizer_Adam():
    def __init__(self, learning_rate=0.001, decay=0., eps=1e-7, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2

    def pre_optimize(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + (self.decay * self.iterations))

    def optimize(self, layer):
        #conditional for initialization (0 array for updated weights)
        if not hasattr(layer, 'weight_cache'):
            layer.weight_memory = np.zeros_like(layer.weights)
            layer.bias_memory = np.zeros_like(layer.bias)

            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.bias)

        #get the weight memory
        layer.weight_memory = (self.beta1 * layer.weight_memory) + (1 - self.beta1) * layer.d_weights
        layer.corrected_weight_memory = layer.weight_memory / (1 - self.beta1 ** (self.iterations + 1))

        #get the bias memory
        layer.bias_memory = (self.beta1 * layer.bias_memory) + (1- self.beta1) * layer.d_bias
        layer.corrected_bias_memory = layer.bias_memory / (1 - self.beta1 ** (self.iterations +1))

        #get the weight change magnitude based on RMSprop
        layer.weight_cache = layer.weight_cache * self.beta2 + ((1 - self.beta2) * (layer.d_weights ** 2))
        layer.corrected_weight_cache = layer.weight_cache / (1 - self.beta2 ** (self.iterations + 1))

        #get the bias change magnitude based on RMSprop
        layer.bias_cache = layer.bias_cache * self.beta2 + ((1- self.beta2) * (layer.d_bias ** 2))
        layer.corrected_bias_cache = layer.bias_cache / (1 - self.beta2 ** (self.iterations + 1))

        layer.weights -= self.current_learning_rate / (np.sqrt(layer.corrected_weight_cache) + self.eps) * layer.corrected_weight_memory
        layer.bias -= self.current_learning_rate / (np.sqrt(layer.corrected_bias_cache) + self.eps) * layer.corrected_bias_memory

    def post_optimize(self):
        self.iterations += 1

class Model():
    def __init__(self):
        self.layer_list = []
        self.weight_layer_list = []
        self.validation_layer_list = []
        self.softmax_cce_backward = None

    def add(self, layer):
        self.layer_list.append(layer)

    def set(self, *, loss_function, accuracy_function, optimizer):
        self.loss_function = loss_function
        self.accuracy_function = accuracy_function
        self.optimizer = optimizer

        print(len(self.layer_list))

    def finalise(self):
        for i in range(len(self.layer_list)):
            if i == 0:
                self.init_layer = Init_layer()
                self.layer_list[i].prev = self.init_layer
                self.layer_list[i].next = self.layer_list[i+1]

            elif i < len(self.layer_list) - 1:
                self.layer_list[i].prev = self.layer_list[i-1]
                self.layer_list[i].next = self.layer_list[i+1]

            else:
                self.layer_list[i].prev = self.layer_list[i-1]
                self.layer_list[i].next = self.loss_function
                self.last_activation_function = self.layer_list[i]

        #set layers with weights
        for layer in self.layer_list:
            if hasattr(layer, 'weights'):
                self.weight_layer_list.append(layer)

        #filter dropout layers for validation pass
        for layer in self.layer_list:
            if not isinstance(layer, Dropout):
                print(isinstance(layer, Dropout))
                self.validation_layer_list.append(layer)

        if isinstance(self.layer_list[-1], Softmax) and isinstance(self.loss_function, CCE_loss):
            self.softmax_cce_backward = Combined_Softamx_and_CCE_loss()
            
    def forward(self, data):
        self.init_layer.forward(data)

        for layer in self.layer_list:
            layer.forward(layer.prev.output)

        return layer.output #layer is now the last object in the list

    def backward(self, input, label):
        #check if softmax and CCE are used
        if self.softmax_cce_backward is not None:
            self.softmax_cce_backward.backward(softmax_outputs=input, labels=label)
            self.layer_list[-1].gradients = self.softmax_cce_backward.gradients

            for layer in reversed(self.layer_list[:-1]):
                layer.backward(layer.next.gradients)

        else:
            #initialize the loss output
            self.loss_function.backward(input, label)

            #loop thought the layer list and run .backward
            for layer in reversed(self.layer_list):
                layer.backward(layer.next.gradients)

    def validate(self, x_val, y_val):
        pass

    def train(self, epochs, data, labels, validation_data=None):
        self.accuracy_function.init(labels)

        for epoch in range(epochs):
            #forward pass
            self.network_output = self.forward(data=data)
            
            #loss calculation
            self.forward_loss, self.reg_loss = self.loss_function.calculate(self.network_output, labels, self.weight_layer_list)
            self.total_loss = self.forward_loss + self.reg_loss

            #accuracy calculation
            self.predictions = self.last_activation_function.prediction(self.network_output)
            self.accuracy = self.accuracy_function.calculate(self.predictions, labels)
            
            #backward pass
            self.backward(input=self.network_output, label=labels)

            #optimization
            self.optimizer.pre_optimize()
            for layer in self.weight_layer_list:
                self.optimizer.optimize(layer)
            self.optimizer.post_optimize()

            print(f'Epoch: {epoch}',
                  f'Total Loss: {self.total_loss}',
                  f'Accuracy: {self.accuracy}',
                  f'lr: {self.optimizer.current_learning_rate}'
                  )
            
        #run validation at the end of the training run
        if validation_data is not None:
            x_val, y_val = validation_data

            #forward pass
            self.val_output = self.forward(data=x_val)

            #loss calculation
            self.val_forward_loss, self.val_reg_loss = self.loss_function.calculate(self.val_output, y_val, self.weight_layer_list)

            #accuracy calculation
            self.val_predictions = self.last_activation_function.prediction(self.val_output)
            self.val_accuracy = self.accuracy_function.calculate(self.val_predictions, y_val)

            print(f'Validation Loss: {self.val_forward_loss}',
                  f'Validation Accuracy: {self.val_accuracy}'
                  )

    def inference(self):
        # #omit the dropoutlayers somehow
        # self.accuracy_function.init(y_val)

        # self.init_layer.forward(x_val)

        # for layer in self.validation_layer_list:
        #     layer.forward(layer.prev.output)

        # return layer.output #layer is now the last object in the list

        # self.network_output = self.forward(data=x_val)
        
        # #loss calculation
        # self.loss = self.loss_function.calculate(self.network_output, labels, self.weight_layer_list)

        # #accuracy calculation
        # self.predictions = self.last_activation_function.prediction(self.network_output)
        # self.accuracy = self.accuracy_function.calculate(self.predictions, labels)
        
        # #backward pass
        # self.backward(input=self.network_output, label=labels)

        # #optimization
        # self.optimizer.pre_optimize()
        # for layer in self.weight_layer_list:
        #     self.optimizer.optimize(layer)
        # self.optimizer.post_optimize()

        # print(f'Epoch: {epoch}',
        #         f'Total Loss: {self.loss}',
        #         f'Accuracy: {self.accuracy}',
        #         f'lr: {self.optimizer.current_learning_rate}'
        #         )
        
        pass

#download the data if data is not already in the download folder
url = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
file = 'fashion_mnist_images.zip'
folder = 'fashion_mnist_images'

#data pre-preocessing
if not os.path.isfile(path=file):
    print('Downlaoding file')
    urllib.request.urlretrieve(url=url, filename=file)

else:
    print('File already downloaded')

with ZipFile(file=file) as zip_images:
    zip_images.extractall(folder)

    
'''BINARY CROSS-ENTROPY REGRESSION'''
# x, y = spiral_data(samples=100, classes=2)

# y = y.reshape(-1, 1)

# #model definition
# model = Model()

# model.add(layer=Layer(nr_inputs=2, nr_neurons=64))
# model.add(layer=ReLU())
# model.add(layer=Dropout(keep_rate=0.9))
# model.add(layer=Layer(nr_inputs=64, nr_neurons=64))
# model.add(layer=ReLU())
# model.add(layer=Dropout(keep_rate=0.9))
# model.add(layer=Layer(nr_inputs=64, nr_neurons=1))
# model.add(layer=Sigmoid())

# model.set(loss_function=BCE_loss(), 
#           accuracy_function=BCE_accuracy(), 
#           optimizer=Optimizer_Adam(learning_rate=0.005, decay=1e-3))

# model.finalise()

# model.train(epochs=10001, data=x, labels=y)

# '''REGRESSION'''
# x, y = sine_data()

# #model definition
# model = Model()

# model.add(layer=Layer(nr_inputs=1, nr_neurons=64))
# model.add(layer=ReLU())
# model.add(layer=Layer(nr_inputs=64, nr_neurons=64))
# model.add(layer=ReLU())
# model.add(layer=Layer(nr_inputs=64, nr_neurons=1))
# model.add(layer=Linear())

# model.set(loss_function=MSE_loss(), 
#           accuracy_function=Regression_accuracy(), 
#           optimizer=Optimizer_Adam(learning_rate=0.005, decay=1e-3))

# model.finalise()

# model.train(epochs=10001, data=x, labels=y)

'''CCE CLASSIFICATION'''
# x, y = spiral_data(samples=100, classes=3)

# x_val, y_val = spiral_data(samples=100, classes=3)

# validation_data = x_val, y_val

# #model definition
# model = Model()

# model.add(layer=Layer(nr_inputs=2, nr_neurons=64))
# model.add(layer=ReLU())
# model.add(layer=Dropout(keep_rate=0.9))
# model.add(layer=Layer(nr_inputs=64, nr_neurons=64))
# model.add(layer=ReLU())
# model.add(layer=Dropout(keep_rate=0.9))
# model.add(layer=Layer(nr_inputs=64, nr_neurons=3))
# model.add(layer=Softmax())

# model.set(loss_function=CCE_loss(), 
#           accuracy_function=CCE_accuracy(label_one_hot=False), 
#           optimizer=Optimizer_Adam(learning_rate=0.001, decay=1e-7, ))

# model.finalise()

# model.train(epochs=10001, data=x, labels=y, validation_data=validation_data)

'''CLASSIFICATION'''
# #define the dataset
# x, y = spiral_data(samples=100, classes=3)

# #network architecture
# l1 = Layer(2,512, l2_lambda_weights=5e-4, l2_lambda_bias=5e-4)
# l1_relu = Activation()
# l1_dropout = Dropout(dropout_rate=0.9)
# l2 = Layer(512,3)
# l2_softmax = Softmax()

# # loss_function = CCE_loss()
# smax_and_cce = Combined_Softamx_and_CCE()

# #optimizer
# optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-5, eps=1e-7, beta1=0.9, beta2=0.999)

# #training loop
# for i in range(10001):

#     #forward pass
#     l1.forward(x)
#     l1_relu.forward(l1.output)
#     l1_dropout.forward(l1_relu.output)

#     l2.forward(l1_dropout.output)
#     smax_and_cce.forward(l2.output, y)

#     #loss
#     data_loss, accracy = smax_and_cce.loss_value, smax_and_cce.accuracy_value
#     regularization_loss = smax_and_cce.loss.regularization_loss(l1) + smax_and_cce.loss.regularization_loss(l2)

#     loss = data_loss + regularization_loss
#     print(f'Data Loss: {data_loss}, Reg Loss: {regularization_loss}, Total Loss: {loss}, Accuracy: {accracy}')

#     #backwards pass
#     smax_and_cce.backward(softmax_outputs=smax_and_cce.output, labels=y)
#     l2.backward(gradients=smax_and_cce.gradients)

#     l1_dropout.backward(gradients=l2.gradients)
#     l1_relu.backward(gradients=l1_dropout.gradients)
#     l1.backward(gradients=l1_relu.gradients)

#     #optimization
#     optimizer.pre_optimize()
#     optimizer.optimize(layer=l1)
#     optimizer.optimize(layer=l2)
#     optimizer.post_optimize()
    
# #model validation
# x_test, y_test = spiral_data(samples=100, classes=3)

# #forward pass
# l1.forward(x_test)
# l1_relu.forward(l1.output)

# l2.forward(l1_relu.output)
# smax_and_cce.forward(l2.output, y_test)

# #loss
# data_loss, accracy = smax_and_cce.loss_value, smax_and_cce.accuracy_value
# regularization_loss = smax_and_cce.loss.regularization_loss(l1) + smax_and_cce.loss.regularization_loss(l2)

# loss = data_loss + regularization_loss
# print(f'Data Loss: {data_loss}, Reg Loss: {regularization_loss}, Total Loss: {loss}, Accuracy: {accracy}')

'''BINARY CROSS-ENTROPY REGRESSION'''
# #define the dataset
# x, y = spiral_data(samples=100, classes=2)

# y = y.reshape(-1, 1)

# #network architecture
# l1 = Layer(2, 64, l2_lambda_weights=5e-4, l2_lambda_bias=5e-4)
# l1_relu = Activation()
# l1_dropout = Dropout(dropout_rate=0.9)
# l2 = Layer(64, 1)
# l2_sigmoid = Sigmoid()

# # loss_function = CCE_loss()
# loss_bce = BCE_loss()

# #optimizer
# optimizer = Optimizer_Adam(decay=5e-7)

# #training loop
# for i in range(10001):

#     #forward pass
#     l1.forward(x)
#     l1_relu.forward(l1.output)
#     #l1_dropout.forward(l1_relu.output)

#     l2.forward(l1_relu.output)
#     l2_sigmoid.forward(l2.output)
#     avg_loss, avg_acc = loss_bce.calculate(input=l2_sigmoid.output, labels=y)

#     #loss
#     data_loss, accracy = avg_loss, avg_acc
#     regularization_loss = loss_bce.regularization_loss(l1) + loss_bce.regularization_loss(l2)

#     loss = data_loss + regularization_loss
#     print(f'Data Loss: {data_loss}, Reg Loss: {regularization_loss}, Total Loss: {loss}, Accuracy: {accracy}')

#     #backwards pass
#     loss_bce.backward(gradients=l2_sigmoid.output, labels=y)
#     l2_sigmoid.backward(gradients=loss_bce.d_loss)
#     l2.backward(gradients=l2_sigmoid.d_sigmoid)

#     #l1_dropout.backward(gradients=l2.gradients)
#     l1_relu.backward(gradients=l2.gradients)
#     l1.backward(gradients=l1_relu.gradients)

#     #optimization
#     optimizer.pre_optimize()
#     optimizer.optimize(layer=l1)
#     optimizer.optimize(layer=l2)
#     optimizer.post_optimize()
    
# #model validation
# x_test, y_test = spiral_data(samples=100, classes=2)

# y_test = y.reshape(-1, 1)

# #forward pass
# l1.forward(x_test)
# l1_relu.forward(l1.output)

# l2.forward(l1_relu.output)
# l2_sigmoid.forward(l2.output)
# avg_loss, avg_acc = loss_bce.calculate(input=l2_sigmoid.output, labels=y_test)

# #loss
# data_loss, accracy = avg_loss, avg_acc
# regularization_loss = loss_bce.regularization_loss(l1) + loss_bce.regularization_loss(l2)

# loss = data_loss + regularization_loss
# print(f'Data Loss: {data_loss}, Reg Loss: {regularization_loss}, Total Loss: {loss}, Accuracy: {accracy}')

'''REGRESSION'''
# #data
# x, y = sine_data()

# #network architecture
# l1 = Layer(1, 64)
# l1_relu = ReLU()
# # l1_dropout = Dropout(dropout_rate=0.9)
# l2 = Layer(64, 64)
# l2_relu = ReLU()
# l3 = Layer(64, 1)
# l3_linear = Linear()

# # loss_function = CCE_loss()
# loss_mse = MSE_loss()

# #optimizer
# optimizer = Optimizer_Adam(learning_rate=0.005, decay=1e-3)

# #training
# for epoch in range(10000):
#     #forward pass
#     l1.forward(inputs=x)
#     l1_relu.forward(l1.output)
#     # l1_dropout.forward(l1_relu.output)

#     l2.forward(l1_relu.output)
#     l2_relu.forward(l2.output)

#     l3.forward(l2_relu.output)
#     l3_linear.forward(l3.output)

#     loss_mse.forward(input=l3_linear.output, label=y)

#     #calculate loss
#     avg_loss, accuracy = loss_mse.calculate(input=l3_linear.output, labels=y)
#     reg_loss = loss_mse.regularization_loss(l1) + loss_mse.regularization_loss(l2) + loss_mse.regularization_loss(l3)
#     total_loss = avg_loss + reg_loss

#     #backward pass
#     loss_mse.backward(input=l3_linear.output, label=y)
#     l3_linear.backward(gradients=loss_mse.d_loss)
#     l3.backward(gradients=l3_linear.d_linear)

#     l2_relu.backward(gradients=l3.gradients)
#     l2.backward(gradients=l2_relu.d_relu)

#     # l1_dropout.backward(gradients=l2.gradients)
#     l1_relu.backward(gradients=l2.gradients)
#     l1.backward(gradients=l1_relu.d_relu)

#     #optimization
#     optimizer.pre_optimize()
#     optimizer.optimize(layer=l1)
#     optimizer.optimize(layer=l2)
#     optimizer.optimize(layer=l3)
#     optimizer.post_optimize()

#     print(f'Data Loss: {avg_loss}, Reg Loss: {reg_loss}, Total Loss: {total_loss}, Accuracy: {accuracy}')

#validation