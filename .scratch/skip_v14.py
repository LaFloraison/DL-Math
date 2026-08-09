import numpy as np

def relu(h): return np.maximum(h, 0.0)

def init_block(depth, width, scale, seed=0):
    rng = np.random.default_rng(seed)
    return [ (rng.normal(0, scale, (width, width)), np.zeros(width),
              rng.normal(0, scale, (width, width)), np.zeros(width)) for _ in range(depth) ]

def forward(x, blocks):
    """Returns list of per-block (h_prev, z1, z2) records."""
    recs = []
    h = x
    for (W1, b1, W2, b2) in blocks:
        z1 = relu(h @ W1.T + b1)
        z2 = z1 @ W2.T + b2
        F = relu(z2)
        recs.append((h, z1, z2))
        h = F + h
    return h, recs

def train(X, y, depth, width, scale, residual, steps=6000, lr=0.01, seed=0):
    rng = np.random.default_rng(seed)
    blocks0 = init_block(depth, width, scale, seed)
    blocks = [(W1.copy(), b1.copy(), W2.copy(), b2.copy()) for (W1, b1, W2, b2) in blocks0]
    Ws0 = [blk[0] for blk in blocks0]      # first-layer weights
    w = rng.normal(0, 0.05, width)
    curve = []
    for t in range(steps):
        hL, recs = forward(X, blocks)
        pred = np.tanh(hL @ w)
        dloss = 2.0 * (pred - y) * (1.0 - pred ** 2)
        w -= lr * (hL.T @ dloss)
        delta = dloss[:, None] * w[None, :]
        for l in range(depth - 1, -1, -1):
            h_prev, z1, z2 = recs[l]
            W1, b1, W2, b2 = blocks[l]
            if residual:
                dF = delta
            else:
                dF = delta * (z2 > 0).astype(float)
            dz2 = dF * (z2 > 0).astype(float)
            gW2 = dz2.T @ z1
            dz1 = (dz2 @ W2) * (z1 > 0).astype(float)
            gW1 = dz1.T @ h_prev
            d_hprev = dz1 @ W1
            if residual:
                d_hprev = d_hprev + delta          # identity path (+I)
            # update
            W1 -= lr * gW1; W2 -= lr * gW2
            b1 -= lr * dz1.sum(axis=0); b2 -= lr * dz2.sum(axis=0)
            blocks[l] = (W1, b1, W2, b2)
            delta = d_hprev
        if t % 2000 == 0:
            curve.append(np.mean((pred - y) ** 2))
    dW0 = np.linalg.norm(blocks[0][0] - blocks0[0][0])
    return [f"{c:.3f}" for c in curve], dW0

rng = np.random.default_rng(3)
X = rng.normal(0, 1, (128, 16))
y = np.sign(np.sin(X[:, 0] * X[:, 1]) + X[:, 2] ** 2 - X[:, 3])
y = (y > 0).astype(float)

for scale, lr in ((0.05, 0.01), (0.1, 0.005)):
    print(f"=== depth=15, scale={scale}, lr={lr}, 6000 steps ===")
    for residual in (False, True):
        curve, dW0 = train(X, y, 15, 16, scale, residual, lr=lr)
        print(f"  residual={residual}: loss {curve} | bottom ||dW0||={dW0:.2e}")
