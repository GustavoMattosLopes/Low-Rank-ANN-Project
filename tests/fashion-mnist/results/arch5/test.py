import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import *
from nn import NeuralNetwork

(X, y), (X_test, y_test) = load_data(data_dir="/home/gustavomattoslopes/Desktop/ufmg/7th/ML/project/data", dataset="fashion-mnist")

model_path = None if len(sys.argv) != 2 else sys.argv[1]

nn = NeuralNetwork.load(model_path)
print(f"{macro_f1(y, nn.predict(X)):.4f}")