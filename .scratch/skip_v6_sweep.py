import numpy as np

def relu(h): return np.maximum(h, 0.0)

def init_net(depth, width, scale, seed=0):
    rng = np.random.default_rng(seed)
    Ws, bs = [], []
    for _ in range(depth):
        Ws.append(rng.normal(0, scale, (width, width)))
        bs.append(np.zeros(width))
    return Ws, bs

def forward(x, Ws, bs, residual):
    hs = [x]; h = x
    for l, (W, b) in enumerate(zip(Ws, bs)):
        pre = h @ W.T + b
        h = relu(pre)
        if residual: h = h + hs[-1]
        hs.append(h)
    return hs

def train(X, y, depth, width, scale, residual, steps=3000, lr=0.01, seed=0):
    rng = np.random.default_rng(seed)
    Ws, bs = init_net(depth, width, scale, seed)
    w = rng.normal(0, 0.05, width)
    ok = True
    for t in range(steps):
        hs = forward(X, Ws, bs, residual)
        logit = hs[-1] @ w
        pred = np.tanh(logit)
        dloss = 2.0 * (pred - y) * (1.0 - pred ** 2)
        w -= lr * (hs[-1].T @ dloss)
        delta = dloss[:, None] * w[None, :]
        for l in range(depth - 1, -1, -1):
            a_prev = hs[l]; a_out = hs[l + 1]
            pre_act = a_out - a_prev if residual else a_out
            der = (pre_act > 0).astype(float)
            delta_pre = delta * der
            Ws[l] = Ws[l] - lr * (delta_pre.T @ a_prev)
            bs[l] = bs[l] - lr * delta_pre.mean(axis=0)
            g_prev = delta_pre @ Ws[l]
            if residual: g_prev = g_prev + delta
            delta = g_prev
        if not np.isfinite(delta).all() or not np.isfinite(Ws[0]).all():
            ok = False
            break
    if not ok: return None
    pred = np.tanh(forward(X, Ws, bs, residual)[-1] @ w)
    return np.mean((pred > 0) == y)

rng = np.random.default_rng(3)
X = rng.normal(0, 1, (64, 16))
y = (rng.random(64) > 0.5).astype(float)

for scale in (0.05, 0.1):
    for lr in (0.005, 0.01, 0.02):
        for depth in (6, 10):
            row = []
            for residual in (False, True):
                acc = train(X, y, depth, width=16, scale=scale, residual=residual, steps=3000, lr=lr)
                row.append(f"{acc if acc is not None else 'NaN':.3f}")
            print(f"scale={scale} lr={lr} depth={depth}: plain={row[0]} residual={row[1]}")
