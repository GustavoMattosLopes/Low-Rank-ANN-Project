from pathlib import Path
import sys
import os
from os.path import join as ospj

import numpy as np
import pandas as pd

## Set up project root for imports
EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(EXPERIMENT_ROOT).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nn import NeuralNetwork
from lora_nn import LoRANeuralNetwork
from experiment_runner import run_experiment
from utils import load_data, train_val_split

## Hyperparameters being grided
SEEDS = [42, 123, 456, 789, 999]
RANKS = [1, 2, 4, 8, 16]

## Fixed hyperparameters
BATCH_SIZE = 64
EPOCHS = 30

results = []
learning_curves = []

experiment_id = 0
TOTAL_EXPERIMENTS = len(SEEDS) * (1 + len(RANKS)) # 1 for full fine-tuning + len(RANKS) for LoRA experiments

(X, y), (X_test, y_test) = load_data(data_dir=ospj(PROJECT_ROOT, "data"), dataset="fashion-mnist")

for seed in SEEDS:

    # Load data and split into train/val sets
    (X_train, y_train), (X_val, y_val) = train_val_split(X, y, seed=seed)
    
    #
    # Full fine-tuning
    #

    experiment_id += 1
    print(f"\nRunning experiment {experiment_id}/{TOTAL_EXPERIMENTS} | Method: full | Seed: {seed}")

    model = NeuralNetwork(
        sizes=[784, 1024, 512, 256, 128, 64, 10],
        seed=seed,
        learning_rate=1e-3
    )

    result, curves = run_experiment(
        experiment_id=experiment_id,
        method="full",
        seed=seed,
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    results.append(result)
    learning_curves.extend(curves)

    #
    # LoRA
    #

    for rank in RANKS:

        experiment_id += 1
        print(f"\nRunning experiment {experiment_id}/{TOTAL_EXPERIMENTS} | Method: lora | Rank: {rank} | Seed: {seed}")
        
        base_model = NeuralNetwork.load(
            ospj(PROJECT_ROOT, "mnist.npz")
        )

        lora_model = LoRANeuralNetwork(
            base_model=base_model,
            rank=rank,
            alpha=2 * rank, # Common heuristic is to set alpha to 2x or 4x the rank, but this can be tuned as well
            learning_rate=1e-3,
            seed=seed
        )

        result, curves = run_experiment(
            experiment_id=experiment_id,
            method="lora",
            seed=seed,
            rank=rank,
            model=lora_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )

        results.append(result)
        learning_curves.extend(curves)

results_df = pd.DataFrame(results)
curves_df = pd.DataFrame(learning_curves)

RESULTS_FOLDER = ospj(EXPERIMENT_ROOT, "results")

os.makedirs(RESULTS_FOLDER, exist_ok=True)

results_df.to_csv(
    ospj(RESULTS_FOLDER, "results.csv"),
    index=False
)

curves_df.to_csv(
    ospj(RESULTS_FOLDER, "learning_curves.csv"),
    index=False
)

print(
    f"Saved {len(results_df)} experiments."
)