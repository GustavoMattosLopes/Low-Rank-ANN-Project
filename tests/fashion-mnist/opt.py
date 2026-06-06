import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import *
from nn import NeuralNetwork


(X_train, y_train), (X_test, y_test) = load_data(data_dir='../../data', dataset='fashion-mnist')


model = f'arch{int(sys.argv[1])}'

if model == 'arch1':
    original_sizes = [784, 512, 128, 10]
elif model == 'arch2':
    original_sizes = [784, 512, 256, 128, 10]
elif model == 'arch3':
    original_sizes = [784, 1024, 512, 256, 128, 10]
elif model == 'arch4':
    original_sizes = [784, 256, 256, 256, 256, 256, 10]
elif model == 'arch5':
    original_sizes = [784, 512, 512, 512, 512, 512, 10]


dummy = NeuralNetwork(sizes=original_sizes)
original_params = dummy.num_parameters()

print(f"params={original_params}")


def build_lowrank_architecture(ratio):
    sizes = [original_sizes[0]]
    activations = []

    for first, second in zip(original_sizes, original_sizes[1:]):
        rank = int(np.ceil(min(first, second) * ratio))
        sizes.extend([rank, second])
        activations.extend([False, True])

    activations.pop()

    return sizes, activations


with open(f'results/{model}/opt_ratio.csv', 'w') as file:
    file.write('ratio,test_loss,test_acc,f1_score,compressed,compression\n')

    for ratio in np.linspace(0.2, 1.0, 9):
        sizes, activations = build_lowrank_architecture(ratio)

        nn = NeuralNetwork(sizes=sizes, activations=activations, learning_rate=1e-3)
        nn.SGD(X_train, y_train, epochs=50)

        test_loss, test_acc = nn.evaluate(X_test, y_test)
        test_f1 = macro_f1(y_test, nn.predict(X_test))

        compressed_params = nn.num_parameters()
        compression = compressed_params / original_params

        file.write(
            f"{ratio:.2f},"
            f"{test_loss:.4f},"
            f"{100*test_acc:.2f},"
            f"{100*test_f1:.2f},"
            f"{compressed_params},"
            f"{100*compression:.2f}\n"
        )