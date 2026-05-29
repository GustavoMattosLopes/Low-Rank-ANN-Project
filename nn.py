import numpy as np


class NeuralNetwork:
    def __init__(self, sizes, activations=None, learning_rate=1e-2, seed=42):
        self.sizes = sizes
        self.layers = len(sizes)
        self.transitions = self.layers-1

        if activations is None:
            activations = [True] * (self.transitions-1)

        assert len(activations) == (self.transitions-1)

        self.non_linear = activations
        self.lr = learning_rate
        self.rng = np.random.default_rng(seed)

        # Parameters
        self.W = []
        self.b = []

        # Gradients
        self.dW = []
        self.db = []

        # Layer inputs
        self.A = []

        for l in range(self.transitions):
            n_in, n_out = sizes[l], sizes[l+1]

            W = self.rng.standard_normal((n_out, n_in)) * np.sqrt(2 / n_in)
            b = np.zeros((n_out, 1))

            self.W.append(W)
            self.b.append(b)


    def ReLU(self, Z):
        return np.maximum(0, Z)


    def softmax(self, Z):
        exp = np.exp(Z)
        return exp / np.sum(exp, axis=0, keepdims=True)


    def forward(self, X):
        self.A = []
        self.A.append(X)

        for l in range(self.transitions-1):
            W = self.W[l]
            b = self.b[l]
            
            X = W @ X + b 
            if self.non_linear[l]:
                X = self.ReLU(X)
            
            self.A.append(X)

        W = self.W[-1]
        b = self.b[-1]

        scores = self.softmax(W @ X + b)
        self.A.append(scores)
        return scores


    def loss(self, scores, y):
        batch_size = y.shape[0]
        return -np.mean(np.log(scores[y, np.arange(batch_size)]))


    def backward(self, y):
        self.dW = []
        self.db = []
        batch_size = y.shape[0]

        Delta = self.A[-1].copy()
        Delta[y, np.arange(batch_size)] -= 1
        Delta /= batch_size

        for l in range(self.transitions-1, -1, -1):
            W = self.W[l]
            A_prev = self.A[l]

            dW = Delta @ A_prev.T
            db = np.sum(Delta, axis=1, keepdims=True)

            self.dW.append(dW)
            self.db.append(db)

            if l > 0:
                Delta = W.T @ Delta
                if self.non_linear[l-1]:
                    Delta[A_prev <= 0] = 0

        self.dW = self.dW[::-1]
        self.db = self.db[::-1]


    def update(self):
        for l in range(self.transitions):
            self.W[l] -= self.lr * self.dW[l]
            self.b[l] -= self.lr * self.db[l]


    def train_step(self, X, y):
        scores = self.forward(X)
        loss = self.loss(scores, y)
        self.backward(y)
        self.update()
        return loss
    

    def gradient_descent(self, X, y, epochs=20):
        for epoch in range(epochs):
            loss = self.train_step(X, y)
            print(f"epoch {epoch} - loss {loss:.4f}")


    def SGD(self, X, y, epochs=20, batch_size=64):
        num_instances = X.shape[1]

        for epoch in range(epochs):
            perm = self.rng.permutation(num_instances)
            X_epoch = X[:, perm]
            y_epoch = y[perm]

            loss = 0.0
            num_batches = 0
            
            for start in range(0, num_instances, batch_size):
                end = start + batch_size
                X_batch = X_epoch[:, start:end]
                y_batch = y_epoch[start:end]

                loss += self.train_step(X_batch, y_batch)
                num_batches += 1

            loss /= num_batches
            print(f"epoch {epoch} - loss {loss:.4f}")


    def predict(self, X):
        scores = self.forward(X)
        return np.argmax(scores, axis=0)


    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


    def evaluate(self, X, y):
        scores = self.forward(X)
        loss = self.loss(scores, y)
        acc = self.accuracy(X, y)
        return loss, acc


    def save(self, path):
        if not path.endswith(".npz"):
            path += ".npz"

        data = {}
        data["sizes"] = np.array(self.sizes)
        data["non_linear"] = np.array(self.non_linear)
        data["lr"] = np.array(self.lr)

        for l in range(self.transitions):
            data[f"W{l}"] = self.W[l]
            data[f"b{l}"] = self.b[l]

        np.savez_compressed(path, **data)


    @classmethod
    def load(cls, path):
        if not path.endswith(".npz"):
            path += ".npz"

        data = np.load(path)
        sizes = data["sizes"].tolist()
        non_linear = data["non_linear"].tolist()
        lr = float(data["lr"])

        model = cls(
            sizes=sizes,
            activations=non_linear,
            learning_rate=lr
        )

        for l in range(model.transitions):
            model.W[l] = data[f"W{l}"]
            model.b[l] = data[f"b{l}"]

        return model