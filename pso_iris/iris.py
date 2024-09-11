from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from model import evaluate_network 
from pso import pso_iris


# Load and prepare the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
encoder = OneHotEncoder()
y_onehot = encoder.fit_transform(y.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Assuming `best_position` contains the optimal weights found by PSO
hidden_layer_size = 6
iris_classifier = pso_iris(30, 50, hidden_layer_size, X_test_scaled, y_test)
accuracy, conf_matrix = evaluate_network(iris_classifier, hidden_layer_size, X_test_scaled, y_test)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)