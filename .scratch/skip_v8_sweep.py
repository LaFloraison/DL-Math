import numpy as np

def relu(h): return np.maximum(h, 0.0)

def init_net(depth, width, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, 0.1, (width, width)) for _ in range(depth)], \
           [np.zeros(width) for _ in range(depth)]

def forward(x, Ws, bs, residual):
    hs = [x]; h = x
    for l, (W, b) in enumerate(zip(Ws, bs)):
        pre = h @ W.T + b
        h = relu(pre)
        if residual: h = h + hs[-1]
        hs.append(h)
    return hs

def train(X, y, depth, width, residual, steps=2000, lr=0.005, seed=0):
    rng = np.random.default_rng(seed)
    Ws0, bs0 = init_net(depth, width, seed)
    Ws = [W.copy() for W in Ws0]; bs = [b.copy() for b in bs0]
    w = rng.normal(0, 0.05, width)
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
            if residual: g_prev = g_prev + delta
            delta = g_prev
        if not np.isfinite(Ws[0]).all():
            return None
    pred = np.tanh(forward(X, Ws, bs, residual)[-1] @ w)
    loss = np.mean((pred - y) ** 2)
    dW0 = np.linalg.norm(Ws[0] - Ws0[0])
    return loss, dW0

rng = np.random.default_rng(3)
X = rng.normal(0, 1, (128, 16))
y = np.sign(np.sin(X[:, 0] * X[:, 1]) + X[:, 2] ** 2 - X[:, 3])
y = (y > 0).astype(float)

for depth in (6, 10, 15):
    for lr in (0.001, 0.003, 0.005):
        row = []
        for residual in (False, True):
            r = train(X, y, depth, 16, residual, lr=lr)
            if r is None:
                row.append("NaN")
            else:
                row.append(f"{r[0]:.3f}/{r[1]:.1e}")
        print(f"depth={depth} lr={lr}: plain(loss/dW0)={row[0]}  residual={row[1]}")
