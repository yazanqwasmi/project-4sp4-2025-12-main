import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Elementwise sigmoid activation."""
    return 1.0 / (1.0 + np.exp(-x))


def load_mnist(csv_path: str):
    """
    Load MNIST CSV with first column = label, remaining 784 columns = pixels.
    Returns:
        X: (N, 784) float32, pixels in [0, 255] range
        y: (N,) int64 labels
    """
    print("Reading MNIST CSV data...")
    df = pd.read_csv(csv_path)

    # First column is label
    y = df.iloc[:, 0].to_numpy(dtype=np.int64)

    # Remaining columns are pixel values 0–255 (keep in original range)
    X = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    
    return X, y


def load_model_weights(model_dir: str):
    """
    Load pre-trained weights and biases from CSV files.
    Expected shapes:
        W_hidden: (512, 784)
        b_hidden: (512, 1)
        W_out:    (10, 512)
        b_out:    (10, 1)
    """
    print("Loading model weights and biases...")

    W_hidden = pd.read_csv(f"{model_dir}/weights_hidden.csv",
                           header=None).to_numpy(dtype=np.float32)
    b_hidden = pd.read_csv(f"{model_dir}/biases_hidden.csv",
                           header=None).to_numpy(dtype=np.float32)
    W_out = pd.read_csv(f"{model_dir}/weights_output.csv",
                        header=None).to_numpy(dtype=np.float32)
    b_out = pd.read_csv(f"{model_dir}/biases_output.csv",
                        header=None).to_numpy(dtype=np.float32)

    # Flatten biases from (512,1) -> (512,) and (10,1) -> (10,)
    b_hidden = b_hidden.reshape(-1)
    b_out = b_out.reshape(-1)

    print(f"W_hidden shape: {W_hidden.shape}")
    print(f"W_out shape:    {W_out.shape}")

    return W_hidden, b_hidden, W_out, b_out


def forward_pass(X: np.ndarray,
                 W_hidden: np.ndarray, b_hidden: np.ndarray,
                 W_out: np.ndarray, b_out: np.ndarray):
    """
    Perform a full forward pass through the network.

    X:        (N, 784)
    W_hidden: (512, 784)
    b_hidden: (512,)
    W_out:    (10, 512)
    b_out:    (10,)
    """
    # Hidden layer: tanh(X @ W1^T + b1)
    hidden_linear = X @ W_hidden.T + b_hidden  # (N, 512) + (512,) -> (N, 512)
    H = np.tanh(hidden_linear)

    # Output layer: sigmoid(H @ W2^T + b2)
    output_linear = H @ W_out.T + b_out        # (N, 10)
    Z = sigmoid(output_linear)

    return Z


def compute_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    logits: (N, 10) after sigmoid
    labels: (N,) integer class labels 0–9
    """
    y_pred = np.argmax(logits, axis=1)
    correct = (y_pred == labels).sum()
    return correct / labels.shape[0]


def visualize_example(X: np.ndarray, labels: np.ndarray, logits: np.ndarray, index: int = 0):
    """
    Optional: visualize one MNIST example and show predicted vs true label.
    """
    image = X[index].reshape(28, 28)
    y_true = labels[index]
    y_pred = int(np.argmax(logits[index]))

    plt.imshow(image, cmap="gray")
    plt.title(f"True: {y_true}, Predicted: {y_pred}")
    plt.axis("off")
    plt.show()


def main():
    # Adjust these paths to match your repo layout.
    mnist_path = "./data/mnist_train.csv"
    model_dir = "./data/model"

    X, y = load_mnist(mnist_path)
    W_hidden, b_hidden, W_out, b_out = load_model_weights(model_dir)

    print("Running forward pass on all samples...")
    logits = forward_pass(X, W_hidden, b_hidden, W_out, b_out)

    acc = compute_accuracy(logits, y)
    print(f"Training set accuracy: {acc * 100:.2f}%")

    # Optional: visualize one example
    # visualize_example(X, y, logits, index=0)


if __name__ == "__main__":
    main()
