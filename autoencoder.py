"""
autoencoder.py — a small numpy autoencoder (Prof. Kulakov's "autoencoder" primitive).

It compresses the agent's whole-maze observation into a short latent code. That latent is
used as the *key* for episodic memory: "have I been in a situation that looked like
this before, and what worked?" — i.e. analogy-by-similarity, which is exactly the
kind of memory the experiment manipulates.

This is a one-hidden-layer autoencoder (ReLU encoder, linear decoder) trained by
mini-batch SGD with Adam. The convolutional autoencoder (CAE) Prof. Kulakov named is
a drop-in upgrade for Phase 2 (swap the encoder for conv layers via PyTorch); the
*role* in the architecture is identical, so Phase 1 keeps it dependency-free in numpy.
"""

import numpy as np


class Autoencoder:
    def __init__(self, in_dim: int, latent_dim: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        # He-ish init
        self.W1 = rng.normal(0, np.sqrt(2.0 / in_dim), (in_dim, latent_dim)).astype(np.float32)
        self.b1 = np.zeros(latent_dim, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / latent_dim), (latent_dim, in_dim)).astype(np.float32)
        self.b2 = np.zeros(in_dim, dtype=np.float32)
        self._adam = {k: [np.zeros_like(getattr(self, k)), np.zeros_like(getattr(self, k))]
                      for k in ("W1", "b1", "W2", "b2")}
        self._t = 0

    def encode(self, X):
        X = np.atleast_2d(X.astype(np.float32))
        return np.maximum(0.0, X @ self.W1 + self.b1)  # ReLU latent

    def _forward(self, X):
        z = np.maximum(0.0, X @ self.W1 + self.b1)
        Xhat = z @ self.W2 + self.b2
        return z, Xhat

    def fit(self, X, epochs: int = 60, batch: int = 128, lr: float = 1e-3, verbose: bool = True):
        X = X.astype(np.float32)
        rng = np.random.default_rng(0)
        n = len(X)
        for ep in range(epochs):
            idx = rng.permutation(n)
            losses = []
            for s in range(0, n, batch):
                xb = X[idx[s:s + batch]]
                z, xhat = self._forward(xb)
                diff = xhat - xb
                loss = float(np.mean(diff ** 2))
                losses.append(loss)
                m = len(xb)
                dxhat = (2.0 / m) * diff
                gW2 = z.T @ dxhat
                gb2 = dxhat.sum(0)
                dz = dxhat @ self.W2.T
                dz[z <= 0] = 0.0  # ReLU grad
                gW1 = xb.T @ dz
                gb1 = dz.sum(0)
                self._adam_step({"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}, lr)
            if verbose and (ep % 15 == 0 or ep == epochs - 1):
                print(f"    AE epoch {ep:3d}  recon MSE {np.mean(losses):.4f}")
        return self

    def _adam_step(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for k, g in grads.items():
            m, v = self._adam[k]
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * (g * g)
            mhat = m / (1 - b1 ** self._t)
            vhat = v / (1 - b2 ** self._t)
            getattr(self, k)[...] -= lr * mhat / (np.sqrt(vhat) + eps)
