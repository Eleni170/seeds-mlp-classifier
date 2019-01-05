# seeds-mlp-classifier
Classification of seeds using a Multilayer Perceptron.

This repository consists of an Artificial Neural Network which classifies wheat into three different varieties: Kama, Rosa and Canadian.

The dataset uses 7 attributes of the kernel of the wheat and has 70 elements for each variety (210 instances), randomly selected for the experiment. The dataset used is added and can be found officially at [this](https://archive.ics.uci.edu/ml/datasets/seeds) link.

The classifier can use as activation functions Linear (purelin), Hyperbolic Tangent (tansig) and Log-Sigmoid (logsig). Also, as training methods, the Multilayer Perceptron can use Gradient Descent, Gradient Descent with momentum, Conjugate Gradient Descent and Levenberg Marquardt. The first two are written by hand in python functions and the other two are implemented with a python library called [NeuPy](http://neupy.com/pages/home.html).

The classifier also gives the ability to the user to input other variables, such as the number of hidden layers, the number of neurons in each layer, the number of epochs, the value of the least mean square error and the training step. For Gradient Descent with or without momentum, training can stop before the number of epochs are reached if the subtraction of error between 2 sequential epochs is under LMSE. Results are plotted in graphs that show the instances in 2-dimensional space (usage of perimeter and area attributes), the targets and Mean Square Error per epoch. 

In order to use the code in the repository just run main.py in python (python main.py).  
