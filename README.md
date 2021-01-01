# nn-lab
***
Neural Networks are very powerfull tools of machine learning which can be used mainly for various **Prediction, Classification, Generative** applications. 

The Neural Networks are **Universal Function Approximators**. 

They can be visualized as **Multiple Logistic Regression Units Drawing Decision Boundaries In Hyper Space** or **Spatial Tranformations i.e. Representaions of Data Various Waays in Hyper Space** or as **Probability Ditribution Approximators**.
 
 This repo. contains implementation of general neural network from scratch in pure python and numpy.
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
3.  sklearn(optional required only to generate spiral data for in spiral.py)

### Running the code:

1. XOR data fitting and visualization:
        
        python xor.py
2.  Spiral data fitting and visualization
        python spiral.py
3.  To build custom network
        
        import nn.nn as nn
        
    Define neural network as:

        net = nn(shape=[in_features, hidden1, hidden2, ..., hiddenN, out_layer],
                 activations=[act_fn for all hidden layers and output layer], viz=False)

    Training Loop:

        # input and targets should be of shape (Batch_size, n)
        for each epoch:
            for each iteration:
                net.zero_grad()
                output, loss = net(input_batch, target_batch)
            net.adam(lr=learning_rate)


      For more refer **XOR.py**
    
## Sample Visualizations:
To be added

## To Do:
Optimizers are implemented as member functions of **nn** class. In future **Optimizers** should be implemented as seperate class which takes **nn parameters** as inputs.
