import numpy as np

# 1. Sigmoid Activation Function
def sigmoid(z):
    """
    Applies the sigmoid activation function element-wise.
    
    Parameters:
        z (np.ndarray): A scalar or NumPy array (vector/matrix).
    
    Returns:
        np.ndarray: Transformed output in range (0, 1).
    """
    return 1 / (1 + np.exp(-z))

# 2. Linear Transformation
def linear_transform(X, w, b):
    """
    Performs the linear transformation Z = Xw + b.
    
    Parameters:
        X (np.ndarray): Input matrix of shape (n, d)
        w (np.ndarray): Weight vector of shape (d, 1)
        b (float): Scalar bias
    
    Returns:
        np.ndarray: Output Z of shape (n, 1)
    """
    return X @ w + b  # Matrix multiplication with broadcasting

# 3. Feedforward Function
def feedforward(X, w, b):
    """
    Computes the output of a single neuron using sigmoid activation.
    
    Parameters:
        X (np.ndarray): Input matrix (n, d)
        w (np.ndarray): Weight vector (d, 1)
        b (float): Bias term
    
    Returns:
        np.ndarray: Output predictions (n, 1)
    """
    z = linear_transform(X, w, b)
    y_hat = sigmoid(z)
    return y_hat

# 4. Example Usage
if __name__ == "__main__":
    # Input features: 3 samples with 2 features each
    X = np.array([[1.0, 2.0],
                  [2.0, 1.0],
                  [-1.0, -2.0]])

    # Weight vector (d = 2): shape (2, 1)
    w = np.array([[0.1],
                  [-0.2]])

    # Bias scalar
    b = 0.0

    # Perform the feedforward computation
    y_hat = feedforward(X, w, b)

    # Output the predictions
    print("Predicted outputs:")
    print(y_hat)
