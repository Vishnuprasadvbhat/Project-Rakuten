import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def train_network_forward_propagation(weights, hidden_layer_size, X_train_data, y_train_data):
    hidden_layer_weights = weights[:4*hidden_layer_size].reshape(4, hidden_layer_size) 
    output_layer_weights = weights[4*hidden_layer_size:].reshape(hidden_layer_size, 3) 
    
    # Forward pass
    hidden_layer_input = np.dot(X_train_data, hidden_layer_weights)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, output_layer_weights)
    output_layer_output = sigmoid(output_layer_input)
    
    # Loss calculation (Mean Squared Error)
    k = y_train_data - output_layer_output
    loss = np.mean(np.square(k))
    return loss

def evaluate_network(weights, hidden_layer_size, X_test, y_test):
    hidden_layer_weights = weights[:4*hidden_layer_size].reshape(4, hidden_layer_size) 
    output_layer_weights = weights[4*hidden_layer_size:].reshape(hidden_layer_size, 3)
    
    # Forward pass
    hidden_layer_input = np.dot(X_test, hidden_layer_weights)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, output_layer_weights)
    output_layer_output = sigmoid(output_layer_input)
    
    # Convert logits to probabilities
    predictions = softmax(output_layer_output)
    
    # Choose the class with the highest probability
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    true_classes = np.asarray(true_classes)

    # Calculate accuracy
    accuracy = accuracy_score(true_classes, predicted_classes)
    
    # Generate a confusion matrix
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    
    return accuracy, conf_matrix