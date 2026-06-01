import numpy as np

from nn import NeuralNetwork

class LoRANeuralNetwork(NeuralNetwork):
    def __init__(
        self,
        base_model,
        rank=8,
        alpha=16,
        learning_rate=None,
        seed=42
    ):
        super().__init__(
            sizes=base_model.sizes,
            activations=base_model.non_linear,
            learning_rate=(
                learning_rate
                if learning_rate is not None
                else base_model.lr
            ),
            seed=seed
        )

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Copy pretrained weights
        self.W = [W.copy() for W in base_model.W]
        self.b = [b.copy() for b in base_model.b]

        self.rng = np.random.default_rng(seed)

        self.A_lora = []
        self.B_lora = []

        self.dA_lora = []
        self.dB_lora = []

        for l in range(self.transitions):
            n_in = self.sizes[l]
            n_out = self.sizes[l + 1]

            A = self.rng.standard_normal(
                (rank, n_in)
            ) * 0.01

            B = np.zeros((n_out, rank))

            self.A_lora.append(A)
            self.B_lora.append(B)
    
    def effective_weight(self, l):
        return (
            self.W[l]
            + self.scale
            * (self.B_lora[l] @ self.A_lora[l])
        )
    
    def forward(self, X):
        self.A = []
        self.A.append(X)

        for l in range(self.transitions - 1):
            W_eff = self.effective_weight(l)

            X = W_eff @ X + self.b[l]

            if self.non_linear[l]:
                X = self.ReLU(X)

            self.A.append(X)

        W_eff = self.effective_weight(self.transitions - 1)

        scores = self.softmax(
            W_eff @ X + self.b[-1]
        )

        self.A.append(scores)

        return scores
    
    def backward(self, y):
        batch_size = y.shape[0]

        self.dA_lora = []
        self.dB_lora = []
        self.db = []

        Delta = self.A[-1].copy()
        Delta[y, np.arange(batch_size)] -= 1
        Delta /= batch_size

        for l in range(self.transitions - 1, -1, -1):
            A_prev = self.A[l]

            dW_eff = Delta @ A_prev.T

            db = np.sum(
                Delta,
                axis=1,
                keepdims=True
            )

            A = self.A_lora[l]
            B = self.B_lora[l]

            dB = self.scale * (dW_eff @ A.T)
            dA = self.scale * (B.T @ dW_eff)

            self.dB_lora.append(dB)
            self.dA_lora.append(dA)
            self.db.append(db)

            if l > 0:
                W_eff = self.effective_weight(l)

                Delta = W_eff.T @ Delta

                if self.non_linear[l - 1]:
                    Delta[self.A[l] <= 0] = 0

        self.dA_lora.reverse()
        self.dB_lora.reverse()
        self.db.reverse()
    
    def update(self):
        for l in range(self.transitions):
            self.A_lora[l] -= (
                self.lr * self.dA_lora[l]
            )

            self.B_lora[l] -= (
                self.lr * self.dB_lora[l]
            )

            # Optional:
            # train biases too
            self.b[l] -= self.lr * self.db[l]