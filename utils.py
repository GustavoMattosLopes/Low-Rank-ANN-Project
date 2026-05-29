import struct
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


def _read_images(images_filepath):
    with open(images_filepath, "rb") as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))

        if magic != 2051:
            raise ValueError(f"Magic number mismatch, expected 2051, got {magic}")

        data = np.frombuffer(file.read(), dtype=np.uint8)

    images = data.reshape(size, rows * cols)
    images = images.astype(np.float32) / 255.0
    return images.T


def _read_labels(labels_filepath):
    with open(labels_filepath, "rb") as file:
        magic, size = struct.unpack(">II", file.read(8))

        if magic != 2049:
            raise ValueError(f"Magic number mismatch, expected 2049, got {magic}")

        labels = np.frombuffer(file.read(), dtype=np.uint8)

    return labels


def load_mnist(data_dir="data"):
    train_images_path = f"{data_dir}/train-images.idx3-ubyte"
    train_labels_path = f"{data_dir}/train-labels.idx1-ubyte"
    test_images_path = f"{data_dir}/t10k-images.idx3-ubyte"
    test_labels_path = f"{data_dir}/t10k-labels.idx1-ubyte"

    X_train = _read_images(train_images_path)
    y_train = _read_labels(train_labels_path)

    X_test = _read_images(test_images_path)
    y_test = _read_labels(test_labels_path)

    return (X_train, y_train), (X_test, y_test)


def train_val_split(X, y, val_size=10000, seed=42):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[1])

    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    return (X[:, train_idx], y[train_idx]), (X[:, val_idx], y[val_idx])


def plot_image(X, y, y_pred=None):
    img = X.reshape(28, 28)

    plt.imshow(img, cmap="gray")

    if y_pred is None:
        title = f"Label: {y}"
    else:
        status = "correct" if y_pred == y else "wrong"
        title = f"Label: {y} | Predicted: {y_pred} ({status})"

    plt.title(title)
    plt.axis("off")
    plt.show()


def plot_roc_curve(model, X, y, filename=None):
    # Probabilities
    scores = model.forward(X).T
    num_classes = scores.shape[1]

    y_bin = label_binarize(y, classes=np.arange(num_classes))

    plt.figure(figsize=(8, 6))

    for c in range(num_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, c], scores[:, c])

        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"Classe {c} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Aleatório")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("Curvas ROC one-vs-rest")

    plt.legend()
    plt.grid(True)

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.show()