# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 20:00:50 2017

"""
from input_functions import *
from neural_network import NeuralNetwork

import pandas as pd
import numpy as np

if __name__=='__main__':                    
    data=pd.read_csv('seeds.data', header=-1).values        
    eisodoi=np.array(data[:,:7])    
    eisodoi=(eisodoi)*(1/np.linalg.norm(eisodoi))
    stoxoi=np.array(data[:,7])                                                   
    stoxoiInBinary=[]   
    for s in range(len(stoxoi)):
        if(stoxoi[s]==1):            
            stoxoiInBinary.append([0,0,1])            
        elif(stoxoi[s]==2):
            stoxoiInBinary.append([0,1,0])            
        elif(stoxoi[s]==3):
            stoxoiInBinary.append([1,0,0])
                              
    NumberOfNeuronsNInEveryLayer=[7]
    activationFunctionInEveryLayer=[""]
           
    for i in range(check_if_int('Posa krufa strwmata tha dhmiourghthoun? :')):
         print('\n---Krufo Strwma ---',(i+1))                    
         NumberOfNeuronsNInEveryLayer.append(check_if_int('Dwse plithos Neurwnwn sto strwma :'))         
         activationFunctionInEveryLayer.append(check_activation_function())       
    
    NumberOfNeuronsNInEveryLayer.append(3)
    print('\n---Strwma Eksodou')    
    activationFunctionInEveryLayer.append(check_activation_function())
    if(activationFunctionInEveryLayer[-1]!='logsig'):        
        for stoxos in stoxoiInBinary:            
            for stoixeio in range(3):
                if(stoxos[stoixeio]==0):
                    stoxos[stoixeio]=-1                                        
    nn=NeuralNetwork(NumberOfNeuronsNInEveryLayer,activationFunctionInEveryLayer)
    
    number=check_correct_value_for_training()
    if(number==1 or number==2):
        nn.BuildNN()
    
    if(number==1):
        size=check_correct_value_for_training_size_of_training()
        if(size==1):        
            nn.inputsAndTargets(eisodoi,stoxoiInBinary,1,1)            
            nn.Gradient_Descent_Training_No_Validating()            
        elif(size==2):            
            nn.inputsAndTargets(eisodoi,stoxoiInBinary,0.5,1)
            nn.Gradient_Descent_Training_No_Validating()            
            print('MSE = ',nn.anaklhsh())
        elif(size==3):
            nn.inputsAndTargets(eisodoi,stoxoiInBinary,0,1)        
            nn.Gradient_Descent_Training_Testing_Validating()            
            print('MSE = ',nn.anaklhsh())
    elif(number==2):
        nn.inputsAndTargets(eisodoi,stoxoiInBinary,1,1)
        nn.Gradient_Descent_With_Momentum()        
    elif(number==3):        
        nn.inputsAndTargets(eisodoi,stoxoiInBinary,0.5,1)
        nn.Conjugate_Gradient()        
    else:
        nn.inputsAndTargets(eisodoi,stoxoiInBinary,0.5,1)
        nn.Levenberg_Marquardt()
        
    nn.plotting(stoxoi)
   
   
    
