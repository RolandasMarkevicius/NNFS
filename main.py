import numpy as np
import matplotlib as plt
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer:
    def __init__(self, nr_inputs, nr_neurons, l1_lambda_weights=0, l1_lambda_bias=0, l2_lambda_weights=0, l2_lambda_bias=0):
        self.weights = 0.01 * np.random.randn(nr_inputs, nr_neurons)
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
            d_l1_weights[self.d_weights <= 0] = -1
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
            d_l1_bias[self.d_bias <= 0] = -1
            self.d_bias += self.l1_lambda_bias * d_l1_bias

        if self.l2_lambda_bias > 0:
            d_l2_bias = self.l2_lambda_bias * 2 * self.bias
            self.d_bias += d_l2_bias

        #return outputs for further backpropagation
        self.gradients = np.dot(gradients, self.weights.T)
   
class Activation:
    def forward(self, input):
        self.inputs = input
        self.output = np.maximum(0, input)

    def backward(self, gradients):
        self.gradients = gradients.copy()
        self.gradients[self.inputs <= 0] = 0 

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
    
    def regularization_loss(self, layer):
        #l1_loss = sum of absolute value of weights/bias times lambda
        
        regularization_loss = 0

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

class Combined_Softamx_and_CCE():
    def __init__(self):
        #initialise the softmax and cat cross entropy classes
        self.softmax = Softmax()
        self.loss = CCE_loss()

    def forward(self, input, labels):
        self.softmax.forward(input=input)
        self.loss_value, self.accuracy_value = self.loss.calculate(input=self.softmax.output, labels=labels)
        self.output = self.softmax.output

    def backward(self, softmax_outputs, labels):
        #combined partial derivative solves to: softmax outputs - true values
        
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
    def __init__(self, learning_rate=1, decay=0.01, eps=1e-7, beta1=0.9, beta2=0.999):
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

#define the dataset
x, y = spiral_data(samples=100, classes=3)

#network architecture
l1 = Layer(2,64, l2_lambda_weights=5e-4, l2_lambda_bias=5e-4)
l1_relu = Activation()
l2 = Layer(64,3)
l2_softmax = Softmax()

# loss_function = CCE_loss()
smax_and_cce = Combined_Softamx_and_CCE()

#optimizer
optimizer = Optimizer_Adam(learning_rate=0.02, decay=5e-7, eps=1e-7, beta1=0.9, beta2=0.999)

#training loop
for i in range(10001):

    #forward pass
    l1.forward(x)
    l1_relu.forward(l1.output)

    l2.forward(l1_relu.output)
    smax_and_cce.forward(l2.output, y)

    #loss
    data_loss, accracy = smax_and_cce.loss_value, smax_and_cce.accuracy_value
    regularization_loss = smax_and_cce.loss.regularization_loss(l1) + smax_and_cce.loss.regularization_loss(l2)

    loss = data_loss + regularization_loss
    print(f'Data Loss: {data_loss}, Reg Loss: {regularization_loss}, Total Loss: {loss}, Accuracy: {accracy}')

    #backwards pass
    smax_and_cce.backward(softmax_outputs=smax_and_cce.output, labels=y)
    l2.backward(gradients=smax_and_cce.gradients)
    l1_relu.backward(gradients=l2.gradients)
    l1.backward(gradients=l1_relu.gradients)

    #optimization
    optimizer.pre_optimize()
    optimizer.optimize(layer=l1)
    optimizer.optimize(layer=l2)
    optimizer.post_optimize()
    
#model validation
x_test, y_test = spiral_data(samples=100, classes=3)

#forward pass
l1.forward(x_test)
l1_relu.forward(l1.output)

l2.forward(l1_relu.output)
smax_and_cce.forward(l2.output, y_test)

#loss
data_loss, accracy = smax_and_cce.loss_value, smax_and_cce.accuracy_value
regularization_loss = smax_and_cce.loss.regularization_loss(l1) + smax_and_cce.loss.regularization_loss(l2)

loss = data_loss + regularization_loss
print(f'Data Loss: {data_loss}, Reg Loss: {regularization_loss}, Total Loss: {loss}, Accuracy: {accracy}')