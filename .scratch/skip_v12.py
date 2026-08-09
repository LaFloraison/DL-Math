import numpy as np

def relu(h): return np.maximum(h, 0.0)

def init_net(depth, width, scale, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, scale, (width, width)) for _ in range(depth)], \
           [np.zeros(width) for _ in range(depth)]

def forward(x, Ws, bs, residual):
    hs = [x]; h = x
    for l, (W, b) in enumerate(zip(Ws, bs)):
        pre = h @ W.T + b
        h = relu(pre)
        if residual: h = h + hs[-1]
        hs.append(h)
    return hs

def train(X, y, depth, width, scale, residual, steps=3000, lr=0.003, clip=5.0, seed=0):
    rng = np.random.default_rng(seed)
    Ws0, bs0 = init_net(depth, width, scale, seed)
    Ws = [W.copy() for W in Ws0]; bs = [b.copy() for b in bs0]
    w = rng.normal(0, 0.05, width)
    curve = []
    for t in range(steps):
        hs = forward(X, Ws, bs, residual)
        logit = hs[-1] @ w
        pred = np.tanh(logit)
        dloss = 2.0 * (pred - y) * (1.0 - pred ** 2)
        w -= lr * (hs[-1].T @ dloss)
        delta = dloss[:, None] * w[None, :]
        for l in range(depth - 1, -1, -1):
            pre_act = hs[l+1] - hs[l] if residual else hs[l+1]
            der = (pre_act > 0).astype(float)
            delta_pre = delta * der
            Ws[l] = Ws[l] - lr * (delta_pre.T @ hs[l])
            bs[l] = bs[l] - lr * delta_pre.mean(axis=0)
            g_prev = delta_pre @ Ws[l]
            if residual:
                g_prev = g_prev + delta
                n = np.linalg.norm(g_prev)
                if n > clip: g_prev = g_prev * (clip / n)
            delta = g_prev
        if t % 1000 == 0:
            curve.append(np.mean((pred - y) ** 2))
    pred = np.tanh(forward(X, Ws, bs, residual)[-1] @ w)
    dW0 = np.linalg.norm(Ws[0] - Ws0[0])
    return [f"{c:.3f}" for c in curve], np.linalg.norm(dW0)

rng = np.random.default_rng(3)
X = rng.normal(0, 1, (128, 16))
y = np.sign(np.sin(X[:, 0] * X[:, 1]) + X[:, 2] ** 2 - X[:, 3])
y = (y > 0).astype(float)

for scale in (0.03, 0.05):
    print(f"=== scale={scale}, depth=15, 3000 steps ===")
    for residual in (False, True):
        curve, dW0 = train(X, y, 15, 16, scale, residual)
        print(f"  residual={residual}: loss curve {curve} | bottom ||dW0||={dW0:.2e}")
