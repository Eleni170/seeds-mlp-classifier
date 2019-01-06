from input_functions import *
from neuron import Neuron
from layer import Layer

import numpy as np
from neupy import algorithms
import neupy.layers as layerNeupy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class NeuralNetwork:
        
    def __init__(self,neuronsInEveryLayer,activationFunctionInEveryLayer):                                
        self.beta=check_if_correct_beta('Dwste vhma ekpaideushs: ')      
        self.epochs=check_if_int('Dwste arithmo epoxwn: ')
        self.errorTolerance=check_if_float('Dwste elaxisto meso tetragwniko sfalma: ')                
        self.NeuronsInEveryLayer=neuronsInEveryLayer
        self.ActivationFunctionInEveryLayer=activationFunctionInEveryLayer                          
                        
    def BuildNN(self):                    
        self.layers=[]                                   
        for i in range(len(self.NeuronsInEveryLayer)): #Dhmiourgia strwmatwn 
            if(i==len(self.NeuronsInEveryLayer)-1):neuronsNextLayer=0
            else:neuronsNextLayer=self.NeuronsInEveryLayer[i+1]
            self.layers.append(Layer(np.array([None]*(self.NeuronsInEveryLayer[i])),self.NeuronsInEveryLayer[i],neuronsNextLayer,self.ActivationFunctionInEveryLayer[i]))                                                 
        for i in range(len(self.layers)-1): #eisagwgh bias se ola ta strwmata plhn ths eksodou          
            self.layers[i].neurons.append(Neuron(np.array([-1]),self.NeuronsInEveryLayer[i+1]))                
        return
    
    def inputsAndTargets(self,x,t,allPatterns,saveErrors):
        self.inputs=x
        self.targets=t
        self.xtrain=x
        self.ttrain=t
        if hasattr(self, 'errors'):
            del self.errors
        if(saveErrors):            
            self.errors=[]        
        if(allPatterns==0.5):                       
            self.xtrain,self.xtest,self.ttrain,self.ttest = train_test_split(x, t, test_size=0.3)                  
        elif(allPatterns==0):            
            self.xtrainGen,self.xtest,self.ttrainGen,self.ttest = train_test_split(x, t, test_size=0.3) 
            self.xtrain,self.xvalidate,self.ttrain,self.tvalidate = train_test_split(self.xtrainGen, self.ttrainGen, test_size=0.15)                                   
        return
    
    def evaluate0and1(self,predict,criterion):
        TrueFalsePositNeg=[]
        for i in range(len(predict)):
            for j in range(len(predict[0])):                
                if(predict[i][j]==0):
                    if(self.ttest[i][j]==0):
                        TrueFalsePositNeg.append('TN')
                    else:
                        TrueFalsePositNeg.append('FN')
                else:
                    if(self.ttest[i][j]==0):
                        TrueFalsePositNeg.append('FP')
                    else:
                        TrueFalsePositNeg.append('TP')  
        TN=float(TrueFalsePositNeg.count('TN')) 
        FN=float(TrueFalsePositNeg.count('FN'))
        TP=float(TrueFalsePositNeg.count('TP')) 
        FP=float(TrueFalsePositNeg.count('FP'))
        pr=0 if((TP+FP)==0) else TP/(TP+FP)             
        rec=0 if((TP+FN)==0) else TP/(TP+FN)
        return{
            'accuracy':0 if((TP+TN+FP+FN)==0) else (TP+TN)/(TP+TN+FP+FN),
            'precision':pr,
            'recall':rec,
            'fmeasure':0 if((pr+rec)==0) else (pr*rec)/((pr+rec)/2),
            'sensitivity':0 if((TP+FN)==0) else TP/(TP+FN),
            'specificity':0 if((TN+FP)==0) else TN/(TN+FP),    
        }[criterion]


    def evaluateMinus1and1(self,predict,criterion):
        TrueFalsePositNeg=[]
        for i in range(len(predict)):
            for j in range(len(predict[0])):                
                if(predict[i][j]==-1):
                    if(self.ttest[i][j]==-1):
                        TrueFalsePositNeg.append('TN')
                    else:
                        TrueFalsePositNeg.append('FN')
                else:
                    if(self.ttest[i][j]==-1):
                        TrueFalsePositNeg.append('FP')
                    else:
                        TrueFalsePositNeg.append('TP')  
        TN=float(TrueFalsePositNeg.count('TN')) 
        FN=float(TrueFalsePositNeg.count('FN'))
        TP=float(TrueFalsePositNeg.count('TP')) 
        FP=float(TrueFalsePositNeg.count('FP'))
        pr=0 if((TP+FP)==0) else TP/(TP+FP)             
        rec=0 if((TP+FN)==0) else TP/(TP+FN)
        return{
            'accuracy':0 if((TP+TN+FP+FN)==0) else (TP+TN)/(TP+TN+FP+FN),
            'precision':pr,
            'recall':rec,
            'fmeasure':0 if((pr+rec)==0) else (pr*rec)/((pr+rec)/2),
            'sensitivity':0 if((TP+FN)==0) else TP/(TP+FN),
            'specificity':0 if((TN+FP)==0) else TN/(TN+FP),    
        }[criterion]
    
    def Gradient_Descent_Training_No_Validating(self):
        JrPlus1=0
        oldweights=[]    
        for layer in range(len(self.layers)-1):
            for neuron in range(len(self.layers[layer].neurons)):
                for weight in range(len(self.layers[layer].neurons[neuron].weights)):
                    oldweights.append(0)               
        for epoch in range(self.epochs):                                                                                               
            TotalError=0.0           
            for protupo in range(len(self.ttrain)):
                sumW=0                                                                                                            
                self.inputsAndOutputsOfEveryNeuron(self.xtrain[protupo])                                
                TotalError=TotalError+self.calculateError(self.ttrain[protupo])                
                index=0                
                for layer in range(len(self.layers)-1):
                    for neuron in range(len(self.layers[layer].neurons)):
                        for weight in range(len(self.layers[layer].neurons[neuron].weights)):                            
                            oldweights[index]=self.layers[layer].neurons[neuron].weights[weight]
                            index+=1                                                                 
                deltas=[] #ta delta twn neurwnwn apothkeuontai sth morfh deltas=[[doutput0,doutpout1,doutput2],[dhidden0,dhidden1,...],...]
                subdeltas=[]                   
                for neuron in range(len(self.layers[-1].neurons)):                                    
                    subdeltas.append((self.ttrain[protupo][neuron]-self.layers[-1].neurons[neuron].outputNeuron)*(self.layers[-1].activationFunction_d(self.layers[-1].neurons[neuron].outputNeuron)))        
                deltas.append(subdeltas)
                d=0
                for layer in range(len(self.layers)-2,0,-1):                       
                    subdeltas=[] 
                    for neuron in range(len(self.layers[layer].neurons)-1): 
                        sumJ=0
                        for neuronNextLayer in range(len(self.layers[layer].neurons[neuron].weights)):
                           sumJ=sumJ+(deltas[d][neuronNextLayer]*self.layers[layer].neurons[neuron].weights[neuronNextLayer])
                        subdeltas.append(self.layers[layer].activationFunction_d(self.layers[layer].neurons[neuron].outputNeuron)*sumJ)            
                    deltas.append(subdeltas)
                    d=d+1
                d=0
                for layer in range(len(self.layers)-2,-1,-1):
                    for neuron in range(len(self.layers[layer].neurons)):                
                        for weight in range(len(self.layers[layer].neurons[neuron].weights)):                                                                
                            self.layers[layer].neurons[neuron].weights[weight]=self.layers[layer].neurons[neuron].weights[weight]+self.beta*(deltas[d][weight]*self.layers[layer].neurons[neuron].outputNeuron)
                    d=d+1    
                index=0                
                for layer in range(len(self.layers)-1):
                    for neuron in range(len(self.layers[layer].neurons)):
                        for weight in range(len(self.layers[layer].neurons[neuron].weights)):
                            sumW=sumW+abs(oldweights[index]-self.layers[layer].neurons[neuron].weights[weight])                               
                            index+=1                                                                                                              
                if(sumW<self.errorTolerance):break                                                                                                                                                                                                       
            Jr=(1.0/len(self.ttrain))*TotalError                               
            if hasattr(self, 'errors'):
                self.errors.append(Jr)
            if(Jr<self.errorTolerance):break
            if(abs(JrPlus1-Jr)<self.errorTolerance):break            
            JrPlus1=Jr            
            if hasattr(self, 'errors'):
                print('epoch=',epoch,'    Jrtrain=',Jr)                               
                                   
        return
                
    
    def Gradient_Descent_Training_Testing_Validating(self):
        JrPlus1=0
        oldweights=[]    
        for layer in range(len(self.layers)-1):
            for neuron in range(len(self.layers[layer].neurons)):
                for weight in range(len(self.layers[layer].neurons[neuron].weights)):
                    oldweights.append(0)               
        for epoch in range(self.epochs):                                                                                          
            TotalError=0.0           
            for protupo in range(len(self.ttrain)):
                sumW=0                                                                                                            
                self.inputsAndOutputsOfEveryNeuron(self.xtrain[protupo])                                
                TotalError=TotalError+self.calculateError(self.ttrain[protupo])                
                index=0                
                for layer in range(len(self.layers)-1):
                    for neuron in range(len(self.layers[layer].neurons)):
                        for weight in range(len(self.layers[layer].neurons[neuron].weights)):                            
                            oldweights[index]=self.layers[layer].neurons[neuron].weights[weight]
                            index+=1                                                                 
                    
                deltas=[] #ta delta twn neurwnwn apothkeuontai sth morfh deltas=[[doutput0,doutpout1,doutput2],[dhidden0,dhidden1,...],...]
                subdeltas=[]                   
                for neuron in range(len(self.layers[-1].neurons)):                                    
                    subdeltas.append((self.ttrain[protupo][neuron]-self.layers[-1].neurons[neuron].outputNeuron)*(self.layers[-1].activationFunction_d(self.layers[-1].neurons[neuron].outputNeuron)))        
                deltas.append(subdeltas)
                d=0
                for layer in range(len(self.layers)-2,0,-1):                       
                    subdeltas=[] 
                    for neuron in range(len(self.layers[layer].neurons)-1): 
                        sumJ=0
                        for neuronNextLayer in range(len(self.layers[layer].neurons[neuron].weights)):
                              sumJ=sumJ+(deltas[d][neuronNextLayer]*self.layers[layer].neurons[neuron].weights[neuronNextLayer])
                        subdeltas.append(self.layers[layer].activationFunction_d(self.layers[layer].neurons[neuron].outputNeuron)*sumJ)            
                    deltas.append(subdeltas)
                    d=d+1
                d=0
                for layer in range(len(self.layers)-2,-1,-1):
                    for neuron in range(len(self.layers[layer].neurons)):                
                        for weight in range(len(self.layers[layer].neurons[neuron].weights)):                                                                
                            self.layers[layer].neurons[neuron].weights[weight]=self.layers[layer].neurons[neuron].weights[weight]+self.beta*(deltas[d][weight]*self.layers[layer].neurons[neuron].outputNeuron)
                    d=d+1    
                    
                index=0                
                for layer in range(len(self.layers)-1):
                    for neuron in range(len(self.layers[layer].neurons)):
                        for weight in range(len(self.layers[layer].neurons[neuron].weights)):
                            sumW=sumW+abs(oldweights[index]-self.layers[layer].neurons[neuron].weights[weight])                               
                            index+=1                                                                                                              
                if(sumW<self.errorTolerance):break                                                                                                                                                                                                       
            Jr=(1.0/len(self.ttrain))*TotalError                               
            if hasattr(self, 'errors'):
                self.errors.append(Jr)
            if(Jr<self.errorTolerance):break
            if(abs(JrPlus1-Jr)<self.errorTolerance):break            
            JrPlus1=Jr
                
            TotalErrorValidate=0.0
            for protupoValidate in range(len(self.tvalidate)):
                self.inputsAndOutputsOfEveryNeuron(self.xvalidate[protupoValidate])
                TotalErrorValidate=TotalErrorValidate+self.calculateError(self.tvalidate[protupoValidate]) 
            Jrvalidate=(1.0/len(self.tvalidate))*TotalErrorValidate
            if(Jrvalidate<=self.errorTolerance):break                                                                                                
            if hasattr(self, 'errors'):
                print('epoch=',epoch,'    Jrtrain=',Jr)
            
        return                
                    
    def Gradient_Descent_With_Momentum(self):               
        self.m=check_if_correct_m('Dwste parametro ormhs: ')                                                                                
        JrPlus1=0
        DwrMinus1=[]
        oldweights=[]
        for layer in range(len(self.layers)-1):
            for neuron in range(len(self.layers[layer].neurons)):
                for weight in range(len(self.layers[layer].neurons[neuron].weights)):
                    oldweights.append(0)
                    DwrMinus1.append(0)                                    
        for epoch in range(self.epochs):                                                         
            TotalError=0.0
            for protupo in range(len(self.targets)):                                                               
                sumW=0 
                self.inputsAndOutputsOfEveryNeuron(self.inputs[protupo])                                   
                TotalError=TotalError+self.calculateError(self.targets[protupo])
                index=0                
                for layer in range(len(self.layers)-2,-1,-1):
                    for neuron in range(len(self.layers[layer].neurons)):
                        for weight in range(len(self.layers[layer].neurons[neuron].weights)):                            
                            oldweights[index]=self.layers[layer].neurons[neuron].weights[weight]
                            index+=1                                             
                self.Updating_Weights_Momentum(protupo,DwrMinus1)
                index=0                
                for layer in range(len(self.layers)-2,-1,-1):
                    for neuron in range(len(self.layers[layer].neurons)):
                        for weight in range(len(self.layers[layer].neurons[neuron].weights)):
                            sumW=sumW+abs(oldweights[index]-self.layers[layer].neurons[neuron].weights[weight])
                            DwrMinus1[index]=self.layers[layer].neurons[neuron].weights[weight]-oldweights[index]
                            index+=1                                                                     
                if(sumW<self.errorTolerance):break                                                                                                                                                                                                                                
            Jr=(1.0/len(self.targets))*TotalError           
            self.errors.append(Jr)   
            if(epoch%100==0):
                print('epoch=',epoch,'    Jr=',Jr)
            if(Jr<self.errorTolerance):break
            if(abs(JrPlus1-Jr)<self.errorTolerance):break            
            JrPlus1=Jr
        return
       
    
    def Updating_Weights_Momentum(self,protupo,D):                 
        deltas=[]
        subdeltas=[]                   
        for neuron in range(len(self.layers[-1].neurons)):                                    
            subdeltas.append((self.targets[protupo][neuron]-self.layers[-1].neurons[neuron].outputNeuron)*(self.layers[-1].activationFunction_d(self.layers[-1].neurons[neuron].outputNeuron)))        
        deltas.append(subdeltas)
        d=0
        for layer in range(len(self.layers)-2,0,-1):                       
            subdeltas=[] 
            for neuron in range(len(self.layers[layer].neurons)-1): 
                sumJ=0
                for neuronNextLayer in range(len(self.layers[layer].neurons[neuron].weights)):
                   sumJ=sumJ+(deltas[d][neuronNextLayer]*self.layers[layer].neurons[neuron].weights[neuronNextLayer])
                subdeltas.append(self.layers[layer].activationFunction_d(self.layers[layer].neurons[neuron].outputNeuron)*sumJ)            
            deltas.append(subdeltas)
            d=d+1
        d=0
        index=0
        for layer in range(len(self.layers)-2,-1,-1):
            for neuron in range(len(self.layers[layer].neurons)):                
                for weight in range(len(self.layers[layer].neurons[neuron].weights)):                                                                
                    self.layers[layer].neurons[neuron].weights[weight]=self.layers[layer].neurons[neuron].weights[weight]+self.beta*(deltas[d][weight]*self.layers[layer].neurons[neuron].outputNeuron)+self.m*D[index] 
                    index=index+1
            d=d+1                                         
        return                    
    
    def Initialize_Connection(self):
        neuronsLayers=self.NeuronsInEveryLayer
        activationFsLayers=self.ActivationFunctionInEveryLayer        
        connectionInit=[layerNeupy.Input(neuronsLayers[0])]      
        for i in range(1,len(self.NeuronsInEveryLayer)):
            if(activationFsLayers[i]=='logsig'):connectionInit.append(layerNeupy.Sigmoid(neuronsLayers[i])) 
            elif(activationFsLayers[i]=='tansig'):connectionInit.append(layerNeupy.Tanh(neuronsLayers[i])) 
            else:connectionInit.append(layerNeupy.Linear(neuronsLayers[i]))
        return connectionInit

    def Conjugate_Gradient(self): #etoimh methodos apo tin ergaleiothiki             
        cgd=algorithms.ConjugateGradient(connection=self.Initialize_Connection(),step=self.beta)
        cgd.fit(self.inputs,self.targets)                                        
        cgd.train(self.xtrain,self.ttrain,epochs=self.epochs,epsilon=self.errorTolerance)                                                        
        for i in cgd.errors:            
            self.errors.append(i)
        print(self.errors)
        predictTest=cgd.predict(self.xtest)         
        self.estimating(predictTest)
        return
        
    def Levenberg_Marquardt(self): #etoimh methodos apo tin ergaleiothiki       
        del self.beta
        lm=algorithms.LevenbergMarquardt(connection=self.Initialize_Connection())                                
        lm.fit(self.inputs,self.targets)
        lm.train(self.inputs,self.targets,epochs=self.epochs,epsilon=self.errorTolerance)                                                   
        for i in lm.errors:            
            self.errors.append(i)
        predictTest=lm.predict(self.xtest)
        self.estimating(predictTest)                                                                                                                 
        return
    

    def estimating(self,predictTest):        
        if(-1 in self.ttest[0]):
            evaluate=self.evaluateMinus1and1
        else:
            evaluate=self.evaluate0and1
        print('Accuracy = ',evaluate(predictTest,'accuracy'))
        print('Precision = ',evaluate(predictTest,'precision'))
        print ('Recall = ',evaluate(predictTest,'recall'))
        print ('Fmeasure = ',evaluate(predictTest,'fmeasure'))
        print ('Sensitivity = ',evaluate(predictTest,'sensitivity'))
        print ('Specificity = ',evaluate(predictTest,'specificity'))  
                           
        return            
                                        
    def inputsAndOutputsOfEveryNeuron(self,currentInput):
        for feature in range(len(currentInput)):
            self.layers[0].neurons[feature].inputNeuron=currentInput[feature]
            self.layers[0].neurons[feature].outputNeuron=self.layers[0].neurons[feature].inputNeuron                                         
        for layer in range(1,len(self.layers)):
            neuronRange=len(self.layers[layer].neurons)
            if(layer!=len(self.layers)-1):neuronRange-=1
            for neuron in range(neuronRange):
                diegersh=0.0
                for neuronPreviousLayer in range(len(self.layers[layer-1].neurons)):                                                          
                    diegersh=diegersh+(self.layers[layer-1].neurons[neuronPreviousLayer].outputNeuron*self.layers[layer-1].neurons[neuronPreviousLayer].weights[neuron])                               
                currentLayer=self.layers[layer]  
                currentLayer.neurons[neuron].inputNeuron=diegersh
                currentLayer.neurons[neuron].outputNeuron=currentLayer.activationFunction(diegersh)
                #print 'Diegersh= ',diegersh,' kai eksodos= ',currentLayer.neurons[neuron].outputNeuron                                                                                
        return
    
    def calculateError(self,t):
        error=0.0        
        for neuron in range(len(self.layers[-1].neurons)): #a[-1] shmainei pare to teleutaio stoixeio apo mia lista, a[-2] to proteleutaio klp
            error=error+(t[neuron]-self.layers[-1].neurons[neuron].outputNeuron)**2
        return error
    
    
    def SetRandomWeights(self):
        for layer in range(len(self.layers)-1):
            neuronsNextLayer=len(self.layers[layer+1].neurons)
            if(layer!=(len(self.layers)-2)):
                neuronsNextLayer=neuronsNextLayer-1
            for neuron in range(len(self.layers[layer].neurons)):                
                for weight in range(neuronsNextLayer):
                   self.layers[layer].neurons[neuron].weights[weight]=random.uniform(-1,1)        
        return        
    
    def anaklhsh(self):
        TotalError=0.0           
        for protupo in range(len(self.ttest)):                                                                                                                           
            self.inputsAndOutputsOfEveryNeuron(self.xtest[protupo])
            outputs=[]
            for i in range(len(self.layers[-1].neurons)):
                outputs.append(self.layers[-1].neurons[i].outputNeuron)            
            TotalError=TotalError+self.calculateError(self.ttest[protupo])
        return (1.0/len(self.ttest))*TotalError
        
    
    def plotting(self,stoxoi):        
                               
        if not hasattr(self, 'xtest'):
            fig, axes = plt.subplots(nrows=3, ncols=1)
            fig.tight_layout()
            ax=plt.subplot(3,1,1)        
            plt.title('Ola ta protupa')    
            plot0=plt.plot(self.inputs[:70,0],self.inputs[:70,1],'b.',label='Kama')
            plot1=plt.plot(self.inputs[70:140,0],self.inputs[70:140,1],'r.',label='Rosa')
            plot2=plt.plot(self.inputs[140:,0],self.inputs[140:,1],'g.',label='Canadian')
            plt.legend(handles=[plot0[0],plot1[0],plot2[0]])
            ax.set_xlabel('Area')
            ax.set_ylabel('Perimeter')                        
            
            ax=plt.subplot(3,1,2)        
            plt.title('Stoxoi')
            plt.plot(range(len(stoxoi)),stoxoi,'k.')
            ax.set_xlabel('Protupo')
            ax.set_ylabel('Stoxos')
                                                                                                                                          
            ax=plt.subplot(3,1,3)
            plt.title('Mean Squared Error Per Epoch')        
            plot1=plt.plot(range(len(self.errors)),self.errors,label='Training Error')
            plt.legend(handles=[plot1[0]])
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE')      
        
        else:
            if not hasattr(self, 'xvalidate'):
                fig, axes = plt.subplots(nrows=3, ncols=1)
                fig.tight_layout()
                ax=plt.subplot(3,1,1)        
                
                plt.title('Ola ta protupa')    
                plot0=plt.plot(self.inputs[:70,0],self.inputs[:70,1],'b.',label='Kama')
                plot1=plt.plot(self.inputs[70:140,0],self.inputs[70:140,1],'r.',label='Rosa')
                plot2=plt.plot(self.inputs[140:,0],self.inputs[140:,1],'g.',label='Canadian')
                plt.legend(handles=[plot0[0],plot1[0],plot2[0]])
                ax.set_xlabel('Area')
                ax.set_ylabel('Perimeter')   
                
                ax=plt.subplot(3,1,2)
                      
                plt.title('Protupa Ekpaideushs')                                                
                plot0=plt.plot(self.xtrain[np.array([item[2]==1 for item in self.ttrain]),0],self.xtrain[np.array([item[2]==1 for item in self.ttrain]),1],'b.',label='Kama')
                plot1=plt.plot(self.xtrain[np.array([item[1]==1 for item in self.ttrain]),0],self.xtrain[np.array([item[1]==1 for item in self.ttrain]),1],'r.',label='Rosa')
                plot2=plt.plot(self.xtrain[np.array([item[0]==1 for item in self.ttrain]),0],self.xtrain[np.array([item[0]==1 for item in self.ttrain]),1],'g.',label='Canadian')                       
                plt.legend(handles=[plot0[0],plot1[0],plot2[0]])
                ax.set_xlabel('Area')
                ax.set_ylabel('Perimeter')
                    
                ax=plt.subplot(3,1,3)
                plt.title('Protupa Anaklhshs')                                       
                plot0=plt.plot(self.xtest[np.array([item[2]==1 for item in self.ttest]),0],self.xtest[np.array([item[2]==1 for item in self.ttest]),1],'b.',label='Kama')
                plot1=plt.plot(self.xtest[np.array([item[1]==1 for item in self.ttest]),0],self.xtest[np.array([item[1]==1 for item in self.ttest]),1],'r.',label='Rosa')
                plot2=plt.plot(self.xtest[np.array([item[0]==1 for item in self.ttest]),0],self.xtest[np.array([item[0]==1 for item in self.ttest]),1],'g.',label='Canadian')
                plt.legend(handles=[plot0[0],plot1[0],plot2[0]])
                ax.set_xlabel('Area')
                ax.set_ylabel('Perimeter')
                
                plt.show(block=True)
                
                fig, axes = plt.subplots(nrows=3, ncols=1)
                fig.tight_layout()
                
                ax=plt.subplot(3,1,1)        
                plt.title('Stoxoi olwn twn protupwn')
                plt.plot(range(len(stoxoi)),stoxoi,'k.')
                ax.set_xlabel('Protupo')
                ax.set_ylabel('Stoxos')
                
                ax=plt.subplot(3,1,2)        
                plt.title('Stoxoi protupwn ekpaideushs')
                
                a=[]
                for i in self.ttrain:
                    if(i[2]==1):
                        a.append(1)
                    elif(i[1]==1):
                        a.append(2)
                    else:
                        a.append(3)
                
                plt.plot(range(len(self.ttrain)),a,'k.')
                ax.set_xlabel('Protupo')
                ax.set_ylabel('Stoxos')
                
                a=[]
                for i in self.ttest:
                    if(i[2]==1):
                        a.append(1)
                    elif(i[1]==1):
                        a.append(2)
                    else:
                        a.append(3)
                
                
                ax=plt.subplot(3,1,3)        
                plt.title('Stoxoi protupwn anaklhshs')
                plt.plot(range(len(self.ttest)),a,'k.')
                ax.set_xlabel('Protupo')
                ax.set_ylabel('Stoxos')
                             
                plt.show(block=True)
                
                fig, axes = plt.subplots(nrows=1, ncols=1)
                fig.tight_layout()
                ax=plt.subplot(1,1,1)
                plt.title('Mean Squared Error Per Epoch')        
                plot1=plt.plot(range(len(self.errors)),self.errors,label='Training Error')                
                plt.legend(handles=[plot1[0]])
                ax.set_xlabel('Epoch')
                ax.set_ylabel('MSE')
            else:                    
                fig, axes = plt.subplots(nrows=4, ncols=1)
                fig.tight_layout()            
                ax=plt.subplot(4,1,1)
                plt.title('Ola ta protupa')    
                plot0=plt.plot(self.inputs[:70,0],self.inputs[:70,1],'b.',label='Kama')
                plot1=plt.plot(self.inputs[70:140,0],self.inputs[70:140,1],'r.',label='Rosa')
                plot2=plt.plot(self.inputs[140:,0],self.inputs[140:,1],'g.',label='Canadian')
                plt.legend(handles=[plot0[0],plot1[0],plot2[0]])
                ax.set_xlabel('Area')
                ax.set_ylabel('Perimeter') 
                                                                       
                ax=plt.subplot(4,1,2)
                          
                plt.title('Protupa Ekpaideushs')                                                
                plot0=plt.plot(self.xtrain[np.array([item[2]==1 for item in self.ttrain]),0],self.xtrain[np.array([item[2]==1 for item in self.ttrain]),1],'b.',label='Kama')
                plot1=plt.plot(self.xtrain[np.array([item[1]==1 for item in self.ttrain]),0],self.xtrain[np.array([item[1]==1 for item in self.ttrain]),1],'r.',label='Rosa')
                plot2=plt.plot(self.xtrain[np.array([item[0]==1 for item in self.ttrain]),0],self.xtrain[np.array([item[0]==1 for item in self.ttrain]),1],'g.',label='Canadian')                       
                plt.legend(handles=[plot0[0],plot1[0],plot2[0]])
                ax.set_xlabel('Area')
                ax.set_ylabel('Perimeter')
                            
                ax=plt.subplot(4,1,3)            
                plt.title('Protupa Epikurwshs')                
                plot0=plt.plot(self.xvalidate[np.array([item[2]==1 for item in self.tvalidate]),0],self.xvalidate[np.array([item[2]==1 for item in self.tvalidate]),1],'b.',label='Kama')
                plot1=plt.plot(self.xvalidate[np.array([item[1]==1 for item in self.tvalidate]),0],self.xvalidate[np.array([item[1]==1 for item in self.tvalidate]),1],'r.',label='Rosa')
                plot2=plt.plot(self.xvalidate[np.array([item[0]==1 for item in self.tvalidate]),0],self.xvalidate[np.array([item[0]==1 for item in self.tvalidate]),1],'g.',label='Canadian')                                                
                plt.legend(handles=[plot0[0],plot1[0],plot2[0]])
                ax.set_xlabel('Area')
                ax.set_ylabel('Perimeter')
    
                ax=plt.subplot(4,1,4)
                plt.title('Protupa Anaklhshs')                                       
                plot0=plt.plot(self.xtest[np.array([item[2]==1 for item in self.ttest]),0],self.xtest[np.array([item[2]==1 for item in self.ttest]),1],'b.',label='Kama')
                plot1=plt.plot(self.xtest[np.array([item[1]==1 for item in self.ttest]),0],self.xtest[np.array([item[1]==1 for item in self.ttest]),1],'r.',label='Rosa')
                plot2=plt.plot(self.xtest[np.array([item[0]==1 for item in self.ttest]),0],self.xtest[np.array([item[0]==1 for item in self.ttest]),1],'g.',label='Canadian')
                plt.legend(handles=[plot0[0],plot1[0],plot2[0]])
                ax.set_xlabel('Area')
                ax.set_ylabel('Perimeter')                        
                plt.show(block=True)
                
                fig, axes = plt.subplots(nrows=4, ncols=1)
                fig.tight_layout()
                
                ax=plt.subplot(4,1,1)        
                plt.title('Stoxoi olwn twn protupwn')
                plt.plot(range(len(stoxoi)),stoxoi,'k.')
                ax.set_xlabel('Protupo')
                ax.set_ylabel('Stoxos')
                
                ax=plt.subplot(4,1,2)        
                plt.title('Stoxoi protupwn ekpaideushs')
                
                a=[]
                for i in self.ttrain:
                    if(i[2]==1):
                        a.append(1)
                    elif(i[1]==1):
                        a.append(2)
                    else:
                        a.append(3)
                
                plt.plot(range(len(self.ttrain)),a,'k.')
                ax.set_xlabel('Protupo')
                ax.set_ylabel('Stoxos')                
                
                ax=plt.subplot(4,1,3)        
                plt.title('Stoxoi protupwn epikurwshs')
                
                a=[]
                for i in self.tvalidate:
                    if(i[2]==1):
                        a.append(1)
                    elif(i[1]==1):
                        a.append(2)
                    else:
                        a.append(3)
                
                plt.plot(range(len(self.tvalidate)),a,'k.')
                ax.set_xlabel('Protupo')
                ax.set_ylabel('Stoxos')
                
                ax=plt.subplot(4,1,4)        
                plt.title('Stoxoi protupwn anaklhshs')
                                
                a=[]
                for i in self.ttest:
                    if(i[2]==1):
                        a.append(1)
                    elif(i[1]==1):
                        a.append(2)
                    else:
                        a.append(3)                                                    
               
                plt.plot(range(len(self.ttest)),a,'k.')
                ax.set_xlabel('Protupo')
                ax.set_ylabel('Stoxos')
                
                plt.show(block=True)
                
                
                fig, axes = plt.subplots(nrows=1, ncols=1)
                fig.tight_layout()
                ax=plt.subplot(1,1,1)
                plt.title('Mean Squared Error Per Epoch')        
                plot1=plt.plot(range(len(self.errors)),self.errors,label='Training Error')                
                plt.legend(handles=[plot1[0]])
                ax.set_xlabel('Epoch')
                ax.set_ylabel('MSE')                                
        plt.show(block=True)          
        return

                                                        
    def displayNN(self):
        for i in range(len(self.layers)):
            print ('Layer= ',i)
            for j in range(len(self.layers[i].neurons)):
                print ('Neuron= ',j)
                print ('Input: ',self.layers[i].neurons[j].inputNeuron)
                print ('Output: ',self.layers[i].neurons[j].outputNeuron)                
                if(i!=len(self.layers)-1):
                    print ('Weights: ',self.layers[i].neurons[j].weights)
        print ('---------------')                
        return
