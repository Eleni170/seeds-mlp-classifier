#restrictions for inputs
def check_if_int(phrase):   
    while True:
        try:
            number=int(input(phrase))
            if number<=0:continue
            return number
        except ValueError:continue

def check_correct_value_for_training():
    phrase='Poia methodos ekpaideushs na xrhsimopoihthei? Dwste mia timh apo tis parakatw:\n'
    phrase=phrase+'1. Aplo Gradient Descent.\n'
    phrase=phrase+'2. Aplo Gradient Descent me ormh.\n'
    phrase=phrase+'3. Conjugate Gradient Descent.\n'
    phrase=phrase+'4. Levenberg Marquardt.' 
    while True:
        try:
            number=int(input(phrase))
            if number!=1 and number!=2 and number!=3 and number!=4:continue
            return number
        except ValueError:continue

def check_correct_value_for_training_size_of_training():
    phrase='Posa protupa tha ekpaideutoun? Dwste mia timh apo tis parakatw:\n'
    phrase=phrase+'1. Ola.\n'
    phrase=phrase+'2. 70% Training kai 30% Testing.\n'
    phrase=phrase+'3. 70% Training (opou 15% Validating) kai 30% Testing.\n' 
    while True:
        try:
            number=int(input(phrase))
            if number!=1 and number!=2 and number!=3:continue
            return number
        except ValueError:continue
            
def check_if_correct_beta(phrase): 
    while True:
        try:
            step=float(input(phrase))
            if step>=0 and step<=1: return step 
        except ValueError:continue

def check_if_correct_m(phrase): 
    while True:
        try:
            step=float(input(phrase))
            if step>=0 and step<1: return step 
        except ValueError:continue    
    
def check_if_float(phrase): 
    while True:
        try:
            number=float(input(phrase))
            return number
        except ValueError:continue
                                       
def check_activation_function():
    while True:                  
        activationF=input('Dwse sunarthsh energopoihshs sto strwma (Apodektes times: tansig/logsig/purelin): ')
        if(activationF=='tansig' or activationF=='logsig' or activationF=='purelin'): return activationF
