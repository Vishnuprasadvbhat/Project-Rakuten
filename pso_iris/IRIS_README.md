<h1> Implementation of Particle Swarm Optimization (PSO) on Iris Dataset </h1>

## Data preprocessing 

This script prepares and evaluates a neural network model for the Iris dataset using Particle Swarm Optimization (PSO). 

<br>

<p align="center">
  <img src="https://raw.githubusercontent.com/Vishnuprasadvbhat/Project-Rakuten/master/images/carbon.png" alt="alt"/>
</p>

<br>

1. **Data Preparation**:
   - The Iris dataset is loaded and target labels are one-hot encoded.
   - The data is split into training and test sets.
   - Features are standardized using `StandardScaler`.

2. **Optimization**:
   - Particle Swarm Optimization (PSO) is used to find the optimal weights for the neural network.

3. **Evaluation**:
   - The `evaluate_network` function assesses the model's performance on the test data.
   - Accuracy and confusion matrix are calculated and printed.

This process provides a comprehensive approach to training and evaluating a neural network model using PSO for the Iris dataset.


## Neural Network Forward Propagation and Evaluation

This Python code implements a simple feedforward neural network with one hidden layer, designed to perform classification tasks. It uses two key phases: training with forward propagation and evaluation on test data

<p>
<strong>Training: Forward Propagation</strong>

The function train_network_forward_propagation() simulates forward propagation through the neural network. It takes the network weights, hidden layer size, and training data as input to compute the output and the loss.

<br>

<p align="center">
  <img src="https://raw.githubusercontent.com/Vishnuprasadvbhat/Project-Rakuten/master/images/carbon.png" alt="alt"/>
</p>


<br>

- **Input Layer to Hidden Layer:**
The input data (X_train_data) is multiplied by the hidden layer weights.
- **Hidden Layer to Output Layer:** The output from the hidden layer is passed through the sigmoid activation and then multiplied by the output layer weights.
- **Loss Calculation:** The loss is calculated using Mean Squared Error (MSE) between the predicted output and the actual labels.


<br>

## Metric Evaluation 

<p align="center">
  <img src="https://raw.githubusercontent.com/Vishnuprasadvbhat/Project-Rakuten/master/images/carbon.png" alt="alt"/>
</p

This code implements a simple feedforward neural network with one hidden layer. The network is trained and evaluated using NumPy, and metrics such as accuracy and confusion matrix are calculated using `sklearn`.

### **Iteration Losses in Particle Swarm Optimization (PSO)**

 <br>
The table below shows the best loss values achieved during the first 30 iterations of the PSO process.

<br>

| Iteration | Best Loss   |
|-----------|-------------|
| 1         | 0.2284      |
| 2         | 0.2243      |
| 3         | 0.2220      |
| 4         | 0.2178      |
| 5         | 0.2084      |
| 6         | 0.2039      |
| 7         | 0.1913      |
| 8         | 0.1793      |
| 9         | 0.1712      |
| 10        | 0.1694      |
| 11        | 0.1694      |
| 12        | 0.1688      |
| 13        | 0.1687      |
| 14        | 0.1680      |
| 15        | 0.1652      |
| 16        | 0.1640      |
| 17        | 0.1597      |
| 18        | 0.1593      |
| 19        | 0.1568      |
| 20        | 0.1560      |
| 21        | 0.1560      |
| 22        | 0.1559      |
| 23        | 0.1559      |
| 24        | 0.1559      |
| 25        | 0.1559      |
| 26        | 0.1559      |
| 27        | 0.1559      |
| 28        | 0.1559      |
| 29        | 0.1559      |
| 30        | 0.1556      |



#### This table illustrates the progressive improvement in the model’s performance over 30 iterations, culminating in a loss value of 0.1556.


## Achieved optimal weights

Optimal Weights Found by PSO

<p align="center">
  <img src="https://raw.githubusercontent.com/Vishnuprasadvbhat/Project-Rakuten/master/images/carbon.png" alt="alt"/>
</p

The array of optimal weights represents the best parameters identified by the Particle Swarm Optimization (PSO) algorithm for your neural network model. These weights are crucial for fine-tuning the model’s performance and achieving accurate classification results on the Iris dataset. Each value corresponds to the weight of a particular connection within the neural network, influencing how the model processes input data to make predictions.


## Accuracy and Confusion Matrix of the Network 

<p align="center">
  <img src="https://raw.githubusercontent.com/Vishnuprasadvbhat/Project-Rakuten/master/images/carbon.png" alt="alt"/>
</p


## Particle Movement 


## **Convergence of Particle in Search Space**




- [About PSO](pso_iris\PSO.README.md)