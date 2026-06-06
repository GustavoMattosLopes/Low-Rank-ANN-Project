import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import *
from nn import NeuralNetwork
from lowrank import LowRankDecorator


(X_train, y_train), (X_test, y_test) = load_data(data_dir='../../data', dataset='mnist')

model_path = None if len(sys.argv) != 2 else sys.argv[1]

try:
    nn = NeuralNetwork.load(model_path)
    print("Loaded saved model.")
except Exception:
    print("Training new model.")

    nn = NeuralNetwork(sizes=[784, 512, 128, 10])
    nn.SGD(X_train, y_train)

    if model_path is not None:
        nn.save(model_path)
        print("Model saved.")


train_loss, train_acc = nn.evaluate(X_train, y_train)
print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")

test_loss, test_acc = nn.evaluate(X_test, y_test)
print(f"Test loss:  {test_loss:.4f} | Test acc:  {test_acc:.4f}")

print(f"Test f1-score:  {macro_f1(y_test, nn.predict(X_test)):.4f}")
plot_roc_curve(nn, X_test, y_test, filename= f"{model_path}.png" if model_path else "roc.png")


with open(f'results/svd_ratio.csv', 'w') as file:
    file.write('ratio,test_loss,test_acc,f1_score,compressed,compression\n')

    for ratio in np.linspace(0.2, 1.0, 17):
        lowrank = LowRankDecorator(nn, ratio=ratio)
        test_loss, test_acc = lowrank.evaluate(X_test, y_test)
        test_f1 = macro_f1(y_test, lowrank.predict(X_test))
        rates = lowrank.compression_rates()

        file.write(
            f"{ratio:.2f},"
            f"{test_loss:.4f},"
            f"{100*test_acc:.2f},"
            f"{100*test_f1:.2f},"
            f"{rates['total_compressed_params']},"
            f"{100*rates['compression_ratio']:.2f}\n"
        )


with open(f'results/svd_energy.csv', 'w') as file:
    file.write('energy,test_loss,test_acc,f1_score,compressed,compression\n')

    for energy in np.linspace(0.2, 1.0, 17):
        lowrank = LowRankDecorator(nn, energy=energy)
        test_loss, test_acc = lowrank.evaluate(X_test, y_test)
        test_f1 = macro_f1(y_test, lowrank.predict(X_test))
        rates = lowrank.compression_rates()

        file.write(
            f"{energy:.2f},"
            f"{test_loss:.4f},"
            f"{100*test_acc:.2f},"
            f"{100*test_f1:.2f},"
            f"{rates['total_compressed_params']},"
            f"{100*rates['compression_ratio']:.2f}\n"
        )