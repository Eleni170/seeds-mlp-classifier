import random

class Neuron:
        
    def __init__(self,inputN,numberOfNeuronsNext):    
        self.inputNeuron=inputN
        self.outputNeuron=inputN        
        if(numberOfNeuronsNext!=0):
            self.weights=[random.uniform(-1,1) for _ in range(numberOfNeuronsNext)]
