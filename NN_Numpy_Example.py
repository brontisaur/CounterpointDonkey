import numpy as np

#Activations:

#define an activation function:

def leaky_ReLU(x_in):
    if x_in < 0:
       x_in = 0.01*x_in

    else:
        x_in = x_in

    return x_in

#define an output activation (sigmoid):

def sigmoid(x_in):
    x_in = 1/(1+np.exp(-x_in))
    return x_in

#define an output activation (softmax):

def softmax(x_in):
    x_in = np.exp(x_in)/(np.sum(np.exp(x_in)))
    return x_in

#Create a network class:

class NNet():
    def create(self, input_size, hidden_size, out_size):
        self.inTensors = [] #Create a list of input tensors
        #Create 
    def init_weights(self):

    def init_bias(self):