import numpy as np


class LowRankDecorator:
    def __init__(self, model, ranks=None, energy=0.9):
        # Trained neural network
        self.model = model

        # SVD of each weight matrix
        self.svd_layers = []

        # Rank used for each layer
        self.ranks = []

        for l, W in enumerate(self.model.W):
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            self.svd_layers.append((U, S, Vt))

            if ranks is not None:
                k = min(ranks[l], np.count_nonzero(S > 1e-9))
            else:
                energy_ratio = np.cumsum(S**2) / np.sum(S**2)
                k = np.searchsorted(energy_ratio, energy) + 1  # rank starts at 1

            self.ranks.append(k)


    def _linear_low_rank(self, l, X):
        U, S, Vt = self.svd_layers[l]
        b = self.model.b[l]
        k = self.ranks[l]

        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]

        return (U_k @ (S_k[:, None] * (Vt_k @ X))) + b


    def forward(self, X):
        for l in range(self.model.transitions-1):
            X = self._linear_low_rank(l, X)
            X = self.model.ReLU(X)

        scores = self.model.softmax(self._linear_low_rank(-1, X))
        return scores


    def predict(self, X):
        scores = self.forward(X)
        return np.argmax(scores, axis=0)


    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


    def evaluate(self, X, y):
        scores = self.forward(X)
        loss = self.model.loss(scores, y)
        acc = self.accuracy(X, y)
        return loss, acc


    def compression_rates(self):
        """
        Returns compression statistics for each layer.
        """
        stats = []

        for l, W in enumerate(self.model.W):
            n, m = W.shape
            _, S, _ = self.svd_layers[l]

            rank = np.count_nonzero(S > 1e-9)
            k = self.ranks[l]

            original = n * m
            compressed = k * (n + m + 1)

            stats.append({
                "layer": l,
                "original_rank": rank,
                "compressed_rank": k,
                "rank_ratio": k / rank,
                "original_params": original,
                "compressed_params": compressed,
                "params_ratio": compressed / original
            })

        return stats


    def __getattr__(self, name):
        """
        Automatically delegates missing attributes
        and methods to the original model.
        """
        return getattr(self.model, name)