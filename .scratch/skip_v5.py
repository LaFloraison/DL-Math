import numpy as np

def init_net(depth, width, seed=0):
    rng = np.random.default_rng(seed)
    Ws, bs = [], []
    for _ in range(depth):
        Ws.append(rng.normal(0, 0.1, (width, width)))
        bs.append(np.zeros(width))
    return Ws, bs

def relu(h):
    return np.maximum(h, 0.0)

def forward(x, Ws, bs, residual):
    hs = [x]
    h = x
    for l, (W, b) in enumerate(zip(Ws, bs)):
        pre = h @ W.T + b
        h = relu(pre)
        if residual:
            h = h + hs[-1]
        hs.append(h)
    return hs

def layer_grad_norms(hs, Ws, residual):
    L = len(Ws)
    delta = np.ones_like(hs[-1])
    norms = []
    for l in range(L - 1, -1, -1):
        a_prev = hs[l]
        a_out = hs[l + 1]
        pre_act = a_out - a_prev if residual else a_out
        der = (pre_act > 0).astype(float)
        delta_pre = delta * der
        g_prev = delta_pre @ Ws[l]
        if residual:
            g_prev = g_prev + delta
        norms.append(np.linalg.norm(g_prev))
        delta = g_prev
    return norms[::-1]

np.set_printoptions(precision=4, suppress=True)

depth, width = 20, 20
x = np.random.default_rng(1).normal(0, 1, width)
print("=== Demo 1: gradient norm reaching each layer (ReLU, depth=20) ===")
for residual in (False, True):
    Ws, bs = init_net(depth, width, seed=0)
    hs = forward(x, Ws, bs, residual)
    norms = layer_grad_norms(hs, Ws, residual)
    selected = [0, 4, 8, 12, 16, 19]
    s = [f"{norms[i]:.4e}" for i in selected]
    print(f"residual={residual}: layer {selected} -> grad norm {s}")

def train(X, y, depth, width, residual, steps=3000, lr=0.05, seed=0):
    rng = np.random.default_rng(seed)
    Ws, bs = init_net(depth, width, seed)
    w = rng.normal(0, 0.1, width)          # trainable readout vector
    curves = []
    for t in range(steps):
        hs = forward(X, Ws, bs, residual)
        logit = hs[-1] @ w
        pred = np.tanh(logit)
        loss = np.mean((pred - y) ** 2)
        if t % 500 == 0:
            curves.append(loss)
        dloss = 2.0 * (pred - y) * (1.0 - pred ** 2)      # (64,)
        w -= lr * (hs[-1].T @ dloss)
        delta = dloss[:, None] * w[None, :]               # dL/d h_L
        for l in range(depth - 1, -1, -1):
            a_prev = hs[l]
            a_out = hs[l + 1]
            pre_act = a_out - a_prev if residual else a_out
            der = (pre_act > 0).astype(float)
            delta_pre = delta * der
            gW = delta_pre.T @ a_prev
            Ws[l] = Ws[l] - lr * gW
            bs[l] = bs[l] - lr * delta_pre.mean(axis=0)
            g_prev = delta_pre @ Ws[l]
            if residual:
                g_prev = g_prev + delta
            delta = g_prev
    hs = forward(X, Ws, bs, residual)
    pred = np.tanh(hs[-1] @ w)
    acc = np.mean((pred > 0) == y)
    return curves, acc

rng = np.random.default_rng(3)
X = rng.normal(0, 1, (64, 20))
y = (rng.random(64) > 0.5).astype(float)

print("\n=== Demo 2: fit random labels, depth=10, 3000 steps ===")
for residual in (False, True):
    curves, acc = train(X, y, depth=10, width=20, residual=residual)
    print(f"residual={residual}: loss curve {[f'{c:.3f}' for c in curves]} -> acc={acc:.3f}")
