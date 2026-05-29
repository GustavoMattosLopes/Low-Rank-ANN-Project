import sys
from utils import *
from nn import NeuralNetwork
from lowrank import LowRankDecorator


(X, y), (X_test, y_test) = load_data(dataset="fashion-mnist")

(X_train, y_train), (X_val, y_val) = train_val_split(X, y)

model_path = None if len(sys.argv) != 2 else sys.argv[1]

try:
    nn = NeuralNetwork.load(model_path)
    print("Loaded saved model.")
except Exception:
    print("Training new model.")

    nn = NeuralNetwork(sizes=[784, 1024, 512, 256, 128, 64, 10])
    nn.SGD(X_train, y_train)

    if model_path is not None:
        nn.save(model_path)
        print("Model saved.")

train_loss, train_acc = nn.evaluate(X_train, y_train)
print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")

val_loss, val_acc = nn.evaluate(X_val, y_val)
print(f"Val loss:   {val_loss:.4f} | Val acc:   {val_acc:.4f}")

test_loss, test_acc = nn.evaluate(X_test, y_test)
print(f"Test loss:  {test_loss:.4f} | Test acc:  {test_acc:.4f}")

plot_roc_curve(nn, X_test, y_test, filename="nn.png")

fashion_labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

for i in range(10):
    X = X_test[:, i:i+1]
    y = y_test[i]
    y_pred = int(np.asarray(nn.predict(X)).squeeze())

    plot_image(X, y, y_pred, class_names=fashion_labels)

# -----------------------------------------------------------------
print("\nSVD low rank results:")

nn = LowRankDecorator(nn, energy=0.8)

train_loss, train_acc = nn.evaluate(X_train, y_train)
print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")

val_loss, val_acc = nn.evaluate(X_val, y_val)
print(f"Val loss:   {val_loss:.4f} | Val acc:   {val_acc:.4f}")

test_loss, test_acc = nn.evaluate(X_test, y_test)
print(f"Test loss:  {test_loss:.4f} | Test acc:  {test_acc:.4f}")

compression_rates = nn.compression_rates()
print(compression_rates)

plot_roc_curve(nn, X_test, y_test, filename="svd.png")

# -----------------------------------------------------------------
print("\nOptimizing low rank results:")

ks = [layer["compressed_rank"] for layer in compression_rates]

nn = NeuralNetwork(sizes=[784, ks[0], 1024, ks[1], 512, ks[2], 256, ks[3], 128, ks[4], 64, ks[5], 10],
                   activations=[False, True, False, True, False, True, False, True, False, True, False])

nn.SGD(X_train, y_train)

train_loss, train_acc = nn.evaluate(X_train, y_train)
print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")

val_loss, val_acc = nn.evaluate(X_val, y_val)
print(f"Val loss:   {val_loss:.4f} | Val acc:   {val_acc:.4f}")

test_loss, test_acc = nn.evaluate(X_test, y_test)
print(f"Test loss:  {test_loss:.4f} | Test acc:  {test_acc:.4f}")

plot_roc_curve(nn, X_test, y_test, filename="opt.png")