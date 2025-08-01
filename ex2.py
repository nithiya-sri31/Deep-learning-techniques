import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic dataset
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize weights and bias
def initialize_weights(n_features):
    w = np.zeros(n_features)
    b = 0
    return w, b

# Compute binary cross-entropy loss
def compute_loss(y, y_hat):
    m = y.shape[0]
    return -np.mean(y * np.log(y_hat + 1e-15) + (1 - y) * np.log(1 - y_hat + 1e-15))

# Train logistic regression model
def train(X, y, lr=0.01, epochs=1000):
    n_samples, n_features = X.shape
    w, b = initialize_weights(n_features)

    for i in range(epochs):
        linear_model = np.dot(X, w) + b
        y_hat = sigmoid(linear_model)

        # Compute gradients
        dw = np.dot(X.T, (y_hat - y)) / n_samples
        db = np.sum(y_hat - y) / n_samples

        # Update parameters
        w -= lr * dw
        b -= lr * db

        # Optional: print loss every 100 epochs
        if i % 100 == 0:
            loss = compute_loss(y, y_hat)
            print(f"Epoch {i}, Loss: {loss:.4f}")

    return w, b

# Make predictions
def predict(X, w, b):
    y_hat = sigmoid(np.dot(X, w) + b)
    return (y_hat >= 0.5).astype(int)

# Train and test
w, b = train(X_train, y_train, lr=0.1, epochs=1000)
y_pred = predict(X_test, w, b)

# Evaluate accuracy
acc = accuracy_score(y_test, y_pred)
print("Custom Logistic Regression Accuracy:", acc)
