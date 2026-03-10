import numpy as np
import matplotlib.pyplot as plt
import nnfs
import os
import urllib
import urllib.request
import cv2
import pickle
import copy
from zipfile import ZipFile

from nnfs.datasets import spiral_data, sine_data


nnfs.init()

class Layer():
    def __init__(self, nr_inputs, nr_neurons, l1_lambda_weights=0, l1_lambda_bias=0, l2_lambda_weights=0, l2_lambda_bias=0):
        self.weights = 0.01 * np.random.randn(nr_inputs, nr_neurons) #does this implementation need to change?
        self.bias = np.zeros((1, nr_neurons))

        self.l1_lambda_weights = l1_lambda_weights
        self.l2_lambda_weights = l2_lambda_weights
        self.l1_lambda_bias = l1_lambda_bias
        self.l2_lambda_bias = l2_lambda_bias

    def forward(self, inputs, train):
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

    def get_parameters(self):
        return self.weights, self.bias
    
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

class Init_layer():
    def forward(self, input, train):
        self.output = input

class Dropout():
    def __init__(self, keep_rate):
        self.keep_rate = keep_rate
    
    def forward(self, inputs, train=True):
        if not train:
            self.output = inputs
        else:
            self.binary_mask = np.random.binomial(1, self.keep_rate, np.shape(inputs))
            self.output = inputs * self.binary_mask / self.keep_rate

    def backward(self, gradients):
        self.gradients = gradients * self.binary_mask / self.keep_rate

class Linear():
    def forward(self, input, train):
        self.output = input

    def backward(self, gradients):
        self.gradients = gradients.copy() #is a copy detrimental here?

    def prediction(self, outputs):
        return outputs

class ReLU():
    def forward(self, input, train):
        self.inputs = input
        self.output = np.maximum(0, input)

    def backward(self, gradients):
        self.gradients = gradients.copy()
        self.gradients[self.inputs <= 0] = 0 

    def prediction(self, output):
        return output

class Softmax():
    def forward(self, input, train):
        negative_exponents = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = negative_exponents / np.sum(negative_exponents, axis=1, keepdims=True)
    
    def prediction(self, output):
        return np.argmax(output, axis=1)

class Sigmoid():
    def forward(self, input, train):
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
        self.softmax_cce_backward = None

    def add(self, layer):
        self.layer_list.append(layer)

    def set(self, *, loss_function=None, accuracy_function=None, optimizer=None):
        if loss_function is not None:
            self.loss_function = loss_function

        if accuracy_function is not None:
            self.accuracy_function = accuracy_function

        if optimizer is not None:
            self.optimizer = optimizer

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

        if isinstance(self.layer_list[-1], Softmax) and isinstance(self.loss_function, CCE_loss):
            self.softmax_cce_backward = Combined_Softamx_and_CCE_loss()
            
    def forward(self, data, train=True):
        self.init_layer.forward(data, train=train)

        for layer in self.layer_list:
            layer.forward(layer.prev.output, train)

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

    def get_parameters(self):
        #get the weights and biases of the model
        model_parameters = []

        for layer in self.weight_layer_list:
            layer_parameters = layer.get_parameters()
            model_parameters.append(layer_parameters)
        
        return model_parameters
    
    def set_parameters(self, model_parameters):
        #itterate through the layers and parameters
        for layer, layer_parameters in zip(self.weight_layer_list, model_parameters):
            weights, biases = layer_parameters
            layer.set_parameters(weights, biases)

    def validation(self, data, labels, batch_size=None):
        #partition the full training set by the batch count
        val_steps = data.shape[0] // batch_size

        #include any data that is not accounted in integer division
        if val_steps * batch_size < data.shape[0]:
            val_steps += 1

        #set a 0 validation step counter
        val_step_count = 0
        self.val_accumulated_loss = 0
        self.val_accumulated_accuracy = 0

        for step in range(val_steps):

            if batch_size is None:
                x_val_batch = data
                y_val_batch = labels

            else:
                x_val_batch = data[step*batch_size:(step+1)*batch_size]
                y_val_batch = labels[step*batch_size:(step+1)*batch_size]

            #forward pass
            self.val_output = self.forward(data=x_val_batch, train=False)

            #loss calculation
            self.val_forward_loss, self.val_reg_loss = self.loss_function.calculate(self.val_output, y_val_batch, self.weight_layer_list)

            #accuracy calculation
            self.val_predictions = self.last_activation_function.prediction(self.val_output)
            self.val_accuracy = self.accuracy_function.calculate(self.val_predictions, y_val_batch)

            #accumulate loss, accuracy and steps
            self.val_accumulated_loss += self.val_forward_loss
            self.val_accumulated_accuracy += self.val_accuracy
            val_step_count += 1

        #average validation loss & accuracy
        self.avg_val_loss = self.val_accumulated_loss / val_step_count
        self.avg_val_accuracy = self.val_accumulated_accuracy / val_step_count

        #print validation loss & accuracy
        print(f'Validation Loss: {self.avg_val_loss}',
            f'Validation Accuracy: {self.avg_val_accuracy}'
            )
        
    def save_parameters(self, path):
        #get parameters
        model_parameters = self.get_parameters()
        
        #save the parameters in a file
        with open(path, 'wb') as f:
            pickle.dump(model_parameters, f)

    def load_parameters(self, path):
        #open the file with the model parameters
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(file=f))

    def save(self, path):
        #make a deep copy of the model
        model_copy = copy.deepcopy(self)

        #remove accumulated loss and accuracy
        model_copy.epoch_accumulated_loss = 0
        model_copy.epoch_accumulated_accuracy = 0
        model_copy.epoch_steps = 0

        #remove input data and reset the gradients
        model_copy.init_layer.__dict__.pop('output', None)
        model_copy.loss_function.__dict__.pop('gradients', None)

        #remove properties from all the layers
        for layer in self.layer_list:
            for property in ['inputs', 'output', 'gradients', 'd_weights', 'd_bias']:
                layer.__dict__.pop(property, None)

        with open(path, 'wb') as f:
            pickle.dump(model_copy, f)

    @staticmethod
    def load(path):
        #load the model from a file in the path
        with open(path, 'rb') as f:
            model = pickle.load(f)

        return model

    def train(self, data, labels, *, batch_size=None, epochs=1, include_reg_loss=False):
        self.accuracy_function.init(labels)

        #partition the full training set by the batch count
        steps = data.shape[0] // batch_size

        #include any data that is not accounted in integer division
        if steps * batch_size < data.shape[0]:
            steps += 1

        #itterate through the partitioned list 
        for epoch in range(epochs):
            #reset the epoch loss, accuracy & step count
            self.epoch_accumulated_loss = 0
            self.epoch_accumulated_accuracy = 0
            self.epoch_steps = 0

            for step in range(steps):

                if batch_size is None:
                    batch_data = data
                    batch_labels = labels

                else:
                    batch_data = data[step*batch_size:(step+1)*batch_size]
                    batch_labels = labels[step*batch_size:(step+1)*batch_size]

                #forward pass
                self.network_output = self.forward(data=batch_data, train=True)
                
                #loss calculation
                self.forward_loss, self.reg_loss = self.loss_function.calculate(self.network_output, batch_labels, self.weight_layer_list)
                self.total_loss = self.forward_loss + self.reg_loss

                #print(self.total_loss)

                #accuracy calculation
                self.predictions = self.last_activation_function.prediction(self.network_output)
                self.accuracy = self.accuracy_function.calculate(self.predictions, batch_labels)
                
                #backward pass
                self.backward(input=self.network_output, label=batch_labels)

                #optimization
                self.optimizer.pre_optimize()
                for layer in self.weight_layer_list:
                    self.optimizer.optimize(layer)
                self.optimizer.post_optimize()
                
                #print the current loss & accuracy
                print(f'Step: {step}',
                    f'Total Loss: {self.total_loss}',
                    f'Accuracy: {self.accuracy}',
                    f'lr: {self.optimizer.current_learning_rate}'
                    )
                
                #accumulate loss & accuracy for the epoch loss & accuracy
                if include_reg_loss:
                    self.epoch_accumulated_loss += self.total_loss
                    self.epoch_accumulated_accuracy += self.accuracy
                else:
                    self.epoch_accumulated_loss += self.forward_loss
                    self.epoch_accumulated_accuracy += self.accuracy
                self.epoch_steps += 1

            #calculate average epoch loss and accuracy
            self.epoch_avg_loss = self.epoch_accumulated_loss / self.epoch_steps
            self.epoch_avg_accuracy = self.epoch_accumulated_accuracy / self.epoch_steps

            #print the current epoch loss and accuracy
            print(f'Current Epoch: {epoch}',
                  f'Current Epoch Loss: {self.epoch_avg_loss}',
                  f'Current Epoch Accuracy: {self.epoch_avg_accuracy}')

    def inference(self, input, batch_size=None):
        #get the number of steps required
        steps = 1
        prediction_list = []

        if batch_size is not None:
            steps = input.shape[0] // batch_size

            #include any data that is not accounted in integer division
            if steps * batch_size < input.shape[0]:
                steps += 1

        #itterate through the steps
        for step in range(steps):

            if batch_size is None:
                batch_data = input

            else:
                batch_data = input[step*batch_size:(step+1)*batch_size]

            #run the forward pass
            prediction = self.forward(data=batch_data, train=False)

            #save the outputs to a list
            prediction_list.append(prediction)

        #reshape the list so that the outputs are a stack
        prediction_list = np.vstack(prediction_list)

        return prediction_list

#add train images and labels - shuffles the data and oputputs training and test sample and label lists
def image_data_loading(data_path):
    x = []
    y = []

    x_test = []
    y_test = []

    for label in os.listdir(os.path.join(data_path, 'train')):
        for sample_path in os.listdir(os.path.join(data_path, 'train', label)):
            image = cv2.imread(os.path.join(data_path, 'train', label, sample_path), cv2.IMREAD_UNCHANGED)

            x.append(image)
            y.append(label)
    print('finished training data processing')

    for label_test in os.listdir(os.path.join(data_path, 'test')):
        for sample_test_path in os.listdir(os.path.join(data_path, 'test', label_test)):
            image_test = cv2.imread(os.path.join(data_path, 'test', label_test, sample_test_path), cv2.IMREAD_UNCHANGED)

            x_test.append(image_test)
            y_test.append(label_test)
        
    print('finished test data processing')

    print(f'training samples: {len(x)}')
    print(f'training lables: {len(y)}')

    print(f'test samples: {len(x_test)}')
    print(f'test labels: {len(y_test)}')

    x = np.array(x).astype('uint8')
    y = np.array(y).astype('uint8')

    x_test = np.array(x_test).astype('uint8')
    y_test = np.array(y_test).astype('uint8')

    keys_train = np.array(range(x.shape[0]))
    keys_test = np.array(range(x_test.shape[0]))

    np.random.shuffle(keys_train)
    np.random.shuffle(keys_test)

    x = x[keys_train]
    y = y[keys_train]

    x_test = x_test[keys_test]
    y_test = y_test[keys_test]

    return x, y, x_test, y_test

#data pre-processing - remap
def data_remapping(data):
    #take the data and resize it
    resized_data = (data.astype(np.float32) - 127.5) / 127.5
    reshape_data = (resized_data.reshape(resized_data.shape[0], -1))
    print(f'Output data shape: {reshape_data.shape}')
    return reshape_data

def interpret_prediction(prediction):
    #index dictionary
    prediction_dictionary = {0: 'T-shirt/top',
                             1: 'Trouser',
                             2: 'Pullover',
                             3: 'Dress',
                             4: 'Coat',
                             5: 'Sandal',
                             6: 'Shirt',
                             7: 'Sneaker',
                             8: 'Bag',
                             9: 'Ankle boot'}
    #run argmax
    prediction_index_list = np.argmax(prediction, keepdims=True)

    if len(prediction_index_list) > 1:
        #loop over predictions
        for prediction_index in prediction_index_list:
            interpreted_prediction = prediction_dictionary[prediction_index]

            #print interpretable output
            print(interpreted_prediction)
    else:
        flat_prediction_index_list = prediction_index_list[0][0]
        interpreted_prediction = prediction_dictionary[flat_prediction_index_list]
        print(interpreted_prediction)

        

#download the data if data is not already in the download folder
url = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
file = 'fashion_mnist_images.zip'
folder = 'fashion_mnist_images'

# #data download
# if not os.path.isfile(path=file):
#     print('Downlaoding file')
#     urllib.request.urlretrieve(url=url, filename=file)

#     with ZipFile(file=file) as zip_images:
#         zip_images.extractall(folder)

# else:
#     print('File already downloaded and extracted')

# x, y, x_test, y_test = image_data_loading(data_path=folder)

# x = data_remapping(x)
# x_test = data_remapping(x_test)

# '''CCE CLASSIFICATION'''
# validation_data = x_test, y_test

# #model definition
# model = Model()

# model.add(layer=Layer(nr_inputs=784, nr_neurons=64))
# model.add(layer=ReLU())
# model.add(layer=Dropout(keep_rate=0.9))
# model.add(layer=Layer(nr_inputs=64, nr_neurons=64))
# model.add(layer=ReLU())
# model.add(layer=Dropout(keep_rate=0.9))
# model.add(layer=Layer(nr_inputs=64, nr_neurons=10))
# model.add(layer=Softmax())

# model.set(loss_function=CCE_loss(), 
#           accuracy_function=CCE_accuracy(label_one_hot=False), 
#           optimizer=Optimizer_Adam(learning_rate=0.001, decay=1e-7, ))

# model.finalise()

# model.train(data=x, labels=y, epochs=5, batch_size=128)

# model.validation(data=x_test, labels=y_test, batch_size=128)

# parameters = model.get_parameters()

# #2nd model for testing parameter loading
# print('working on 2nd model')
# model2 = Model()

# model2.add(layer=Layer(nr_inputs=784, nr_neurons=64))
# model2.add(layer=ReLU())
# model2.add(layer=Dropout(keep_rate=0.9))
# model2.add(layer=Layer(nr_inputs=64, nr_neurons=64))
# model2.add(layer=ReLU())
# model2.add(layer=Dropout(keep_rate=0.9))
# model2.add(layer=Layer(nr_inputs=64, nr_neurons=10))
# model2.add(layer=Softmax())

# model2.set(loss_function=CCE_loss(), 
#           accuracy_function=CCE_accuracy(label_one_hot=False))

# model2.finalise()

# model2.set_parameters(model_parameters=parameters)

# model2.validation(data=x_test, labels=y_test, batch_size=128)
# model2.validation(data=x_test, labels=y_test, batch_size=128)

# model2.save_parameters('model_parameters.parms')

# model2.save('fashion_mnist.model')

#sample image
sample_image_path = 'tshirt.png'

#load the image as greyscale 1D
sample_image = cv2.imread(sample_image_path, 0)

downsized_image = cv2.resize(sample_image, dsize=(28, 28))

inverted_image = 255 - downsized_image.astype(np.float32)

resized_iamge = (inverted_image.astype(np.float32) - 127.5) / 127.5

reshaped_image = (resized_iamge.reshape(1, -1))

#load the model
model3 = Model.load(path='fashion_mnist.model')

#run the model for inference
prediction = model3.inference(reshaped_image, 1)

prediction_interpretation = interpret_prediction(prediction=prediction)

#introduce a dictionary to filter the outputs
#use argmax to get the index of the highest predicted output



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