import numpy as np


class LowRankDecorator:
    def __init__(self, model, ranks=None, energy=None, ratio=None):
        # Trained neural network
        self.model = model

        strategies = [ranks is not None, energy is not None, ratio is not None]
        if sum(strategies) != 1:
            raise ValueError("Specify exactly one of: ranks, energy, ratio")

        # SVD of each weight matrix
        self.svd_layers = []

        # Rank used for each layer
        self.ranks = []

        for l, W in enumerate(self.model.W):
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            self.svd_layers.append((U, S, Vt))
            rank = np.count_nonzero(S > 1e-9)

            if ranks is not None:
                k = min(ranks[l], rank)
            elif energy is not None:
                energy_ratio = np.cumsum(S**2) / np.sum(S**2)
                k = np.searchsorted(energy_ratio, energy) + 1  # rank starts at 1
            elif ratio is not None:
                k = max(1, int(np.ceil(ratio * rank)))

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
        stats = []
        total_original = 0
        total_compressed = 0

        for l, W in enumerate(self.model.W):
            n, m = W.shape
            _, S, _ = self.svd_layers[l]

            rank = np.count_nonzero(S > 1e-9)
            k = self.ranks[l]

            original = n * m
            compressed = k * (n + m)

            total_original += original
            total_compressed += compressed

            stats.append({
                "layer": l,
                "original_rank": rank,
                "compressed_rank": k,
                "original_params": original,
                "compressed_params": compressed,
            })

        total_ratio = total_compressed / total_original

        return {
            "layers": stats,
            "total_original_params": total_original,
            "total_compressed_params": total_compressed,
            "compression_ratio": total_ratio,
        }


    def __getattr__(self, name):
        return getattr(self.model, name)