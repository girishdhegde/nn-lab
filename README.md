# nn-lab
***
Implementation of general neural network from scratch in pure python and numpy. Neural Networks are very powerfull tools of machine learning which can be used mainly for various **Prediction, Classification, Generative** applications. The Neural Networks are **Universal Function Approximators**. They can be visualized as **Multiple Logistic Regression Units Drawing Decision Boundaries In Hyper Space** or **Spatial Tranformations i.e. Representaions of Data Various Waays in Hyper Space** or as **Probability Ditribution Approximators**.

***
## Features:
***

1.  Modular: Seperate **neuron, layer, nn** classes with their own **forward** and **backward** functions:
2.  General: NN's of any custom shape can be created
3.  Supports:
    
    *  Activations: Linear, **ReLU**, Sigmoid, Softmax ([Why we need activations](https://stackoverflow.com/a/63543274/14108734))
    
    *  Loss: MSELoss, BCELoss, CrossEntropy/NLL Loss
    
    *  [Optimizers](https://github.com/girishdhegde/optimizers): SGD, Momentum SGD, RMSprop, **Adam**
4.  Supports **Visualization**:
    
    *  NN layers as spatial **Transformations**(Only layers with 2 or 3 layers are supported for visualization)
     
    *  **Decision Boundary**
    
 
## Here's How To Run The Code:
***
### Requirements:
1.  numpy
2.  matplotlib(For visualization if required)
3.  sklearn(optional required only to generate spiral data for testing)

### Running the code:

1. XOR data fitting and visualization:
        
        python xor.py
2.  Spiral data fitting and visualization
        python spiral.py
3.  To build custom network
        
        import nn.nn as nn
    
    And refer **XOR.py**
    
## Sample Visualizations:
To be added
