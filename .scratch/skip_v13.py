import numpy as np

def relu(h): return np.maximum(h, 0.0)

def init_block(depth, width, scale, seed=0):
    """Two-layer residual blocks: h' = relu(W2 relu(W1 h + b1) + b2) + h."""
    rng = np.random.default_rng(seed)
    return [ (rng.normal(0, scale, (width, width)), np.zeros(width),
              rng.normal(0, scale, (width, width)), np.zeros(width)) for _ in range(depth) ]

def forward(x, blocks):
    hs = [x]; h = x
    for (W1, b1, W2, b2) in blocks:
        F = relu(W2 @ relu(W1 @ h + b1) + b2)     # block function F
        h = F + h
        hs.append(h)
    return hs

def train(X, y, depth, width, scale, residual, steps=3000, lr=0.01, clip=5.0, seed=0):
    rng = np.random.default_rng(seed)
    if residual:
        blocks0 = init_block(depth, width, scale, seed)
        blocks = [(W1.copy(), b1.copy(), W2.copy(), b2.copy()) for (W1, b1, W2, b2) in blocks0]
        W1s = [blk[0] for blk in blocks]
    else:
        blocks0 = init_block(depth, width, scale, seed)
        # plain uses only W1 per layer (one linear + relu)
        Ws0 = [blk[0] for blk in blocks0]; bs0 = [blk[1] for blk in blocks0]
        Ws = [W.copy() for W in Ws0]; bs = [b.copy() for b in bs0]
    w = rng.normal(0, 0.05, width)
    curve = []
    for t in range(steps):
        if residual:
            hs = forward(X, blocks)
            logit = hs[-1] @ w
            pred = np.tanh(logit)
            dloss = 2.0 * (pred - y) * (1.0 - pred ** 2)
            w -= lr * (hs[-1].T @ dloss)
            delta = dloss[:, None] * w[None, :]
            for l in range(depth - 1, -1, -1):
                W1, b1, W2, b2 = blocks[l]
                a_prev = hs[l]
                z1 = relu(W1 @ a_prev + b1)          # (width,)
                F = W2 @ z1 + b2
                hF = relu(F)
                out = hF + a_prev
                # dL/d out = delta
                d_hF = delta                              # through identity: +I
                d_F = d_hF * (F > 0).astype(float)
                gW2 = d_F[:, None] * z1[None, :]
                d_z1 = (W2.T @ d_F.T).T
                d_a1 = d_z1 * (z1 > 0).astype(float)
                gW1 = d_a1[:, None] * a_prev[None, :]
                W1 -= lr * gW1; W2 -= lr * gW2
                b1 -= lr * d_a1.mean(axis=0); b2 -= lr * d_F.mean(axis=0)
                blocks[l] = (W1, b1, W2, b2)
                delta = d_a1 @ W1
                n = np.linalg.norm(delta)
                if n > clip: delta = delta * (clip / n)
            dW0 = np.linalg.norm(blocks[0][0] - blocks0[0][0])
        else:
            hs = [X]; h = X
            for (W, b) in zip(Ws, bs):
                h = relu(h @ W.T + b)
                hs.append(h)
            logit = hs[-1] @ w
            pred = np.tanh(logit)
            dloss = 2.0 * (pred - y) * (1.0 - pred ** 2)
            w -= lr * (hs[-1].T @ dloss)
            delta = dloss[:, None] * w[None, :]
            for l in range(depth - 1, -1, -1):
                pre_act = hs[l+1]
                der = (pre_act > 0).astype(float)
                delta_pre = delta * der
                Ws[l] = Ws[l] - lr * (delta_pre.T @ hs[l])
                bs[l] = bs[l] - lr * delta_pre.mean(axis=0)
                delta = delta_pre @ Ws[l]
            dW0 = np.linalg.norm(Ws[0] - Ws0[0])
        if t % 1000 == 0:
            curve.append(np.mean((pred - y) ** 2))
    return [f"{c:.3f}" for c in curve], dW0

rng = np.random.default_rng(3)
X = rng.normal(0, 1, (128, 16))
y = np.sign(np.sin(X[:, 0] * X[:, 1]) + X[:, 2] ** 2 - X[:, 3])
y = (y > 0).astype(float)

for scale, lr in ((0.05, 0.01), (0.05, 0.03)):
    print(f"=== depth=15, scale={scale}, lr={lr}, 3000 steps ===")
    for residual in (False, True):
        curve, dW0 = train(X, y, 15, 16, scale, residual, lr=lr)
        print(f"  residual={residual}: loss curve {curve} | bottom ||dW0||={dW0:.2e}")
