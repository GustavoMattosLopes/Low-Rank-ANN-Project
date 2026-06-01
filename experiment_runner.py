import time
import pandas as pd
import numpy as np

from utils import macro_f1

def run_experiment(
    experiment_id,
    method,
    seed,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=30,
    batch_size=64,
    rank=None, # LoRA rank (Specific to LoRA experiments can be unconsidered for other methods)
):
    """"
     Runs a training experiment for a given model and dataset, returning the results and learning curves.
     params:
        experiment_id: Unique identifier for the experiment (int)
        method: Name of the method being evaluated (str) Exaples: ["method_1", "method_2", "lora", ...]
        seed: Random seed used for reproducibility (int)
        model: The neural network model to be trained (NeuralNetwork or LoRANeuralNetwork instance)
        X_train: Training data features (numpy array of shape [num_features, num_train_samples])
        y_train: Training data labels (numpy array of shape [num_train_samples])
        X_val: Validation data features (numpy array of shape [num_features, num_val_samples])
        y_val: Validation data labels (numpy array of shape [num_val_samples])
        epochs: Number of training epochs (int)
        batch_size: Size of training batches (int)
        rank: Rank used for LoRA models (int or None)
     returns:
        result: Dictionary containing experiment results and metrics
        curves: List of dictionaries containing learning curve data for each epoch
    """

    start = time.perf_counter()

    train_losses, val_losses = model.SGD(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size
    )

    training_time = (
        time.perf_counter() - start
    )

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_acc = np.mean(
        train_pred == y_train
    )

    val_acc = np.mean(
        val_pred == y_val
    )

    train_f1 = macro_f1(
        y_train,
        train_pred
    )

    val_f1 = macro_f1(
        y_val,
        val_pred
    )

    best_epoch = int(
        np.argmin(val_losses)
    )

    result = {
        "experiment_id": experiment_id,
        "method": method,
        "seed": seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "rank": np.nan if rank is None else rank,
        "best_epoch": best_epoch,
        "best_val_loss": np.min(val_losses),
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_f1": train_f1,
        "val_f1": val_f1,
        "training_time_seconds": training_time
    }

    curves = []

    for epoch, (train_loss, val_loss) in enumerate(
        zip(train_losses, val_losses)
    ):
        curves.append({
            "experiment_id": experiment_id,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

    return result, curves