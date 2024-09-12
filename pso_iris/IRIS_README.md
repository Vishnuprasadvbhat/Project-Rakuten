<h1> Implementation of Particle Swarm Optimization (PSO) on Iris Dataset </h1>

This Python code implements a simple feedforward neural network with one hidden layer, designed to perform classification tasks. It uses two key phases: training with forward propagation and evaluation on test data

# Neural Network Forward Propagation and Evaluation


<p>
<strong>Training: Forward Propagation</strong>

The function train_network_forward_propagation() simulates forward propagation through the neural network. It takes the network weights, hidden layer size, and training data as input to compute the output and the loss.

Input Layer to Hidden Layer: The input data (X_train_data) is multiplied by the hidden layer weights.
Hidden Layer to Output Layer: The output from the hidden layer is passed through the sigmoid activation and then multiplied by the output layer weights.
Loss Calculation: The loss is calculated using Mean Squared Error (MSE) between the predicted output and the actual labels.</p>


<p align="center">
  <img src="" alt="alt"/>
</p>



This code implements a simple feedforward neural network with one hidden layer. The network is trained and evaluated using NumPy, and metrics such as accuracy and confusion matrix are calculated using `sklearn`.

```python
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Softmax function to convert logits to probabilities
def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

```

``` python
    # Calculate accuracy
    accuracy = accuracy_score(true_classes, predicted_classes)
    
    # Generate a confusion matrix
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    
    return accuracy, conf_matrix
  ```

- [About PSO](pso_iris\PSO.README.md)