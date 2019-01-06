import numpy as np
from neuron import Neuron

#activation functions
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
    
def sigmoid_d(x):
    return x*(1.0-x)
    
def tanh(x):
    return np.tanh(x)   

def tanh_d(x):
    return 1.0 - x**2 
    
def linear(x):
    return x 
    
def linear_d(_):
    return 1

class Layer:
    
    def __init__(self,x,numberOfNeuronsCurrent,numberOfNeuronsNext,act):                               
        if(act=='logsig'):
            self.activationFunction=sigmoid
            self.activationFunction_d=sigmoid_d
        elif(act=='tansig'):
            self.activationFunction=tanh
            self.activationFunction_d=tanh_d
        elif(act=='purelin'):
            self.activationFunction=linear
            self.activationFunction_d=linear_d
        else:
            self.activationFunction=None
            self.activationFunction_d=None
        self.neurons=[]
        for i in range(numberOfNeuronsCurrent):  #eisagwgh neurwnwn sto strwma              
            self.neurons.append(Neuron(x[i],numberOfNeuronsNext))
