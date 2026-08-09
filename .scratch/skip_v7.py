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

def layer_grad_norms(hs, Ws, residual):
    L = len(Ws)
    delta = np.ones_like(hs[-1])
    norms = []
    for l in range(L - 1, -1, -1):
        pre_act = hs[l+1] - hs[l] if residual else hs[l+1]
        der = (pre_act > 0).astype(float)
        delta_pre = delta * der
        g_prev = delta_pre @ Ws[l]
        if residual: g_prev = g_prev + delta
        norms.append(np.linalg.norm(g_prev))
        delta = g_prev
    return norms[::-1]

np.set_printoptions(precision=4, suppress=True)

# ===== Demo 1: gradient flow (Figure 9.3) =====
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

# ===== Demo 2: do bottom layers actually learn? =====
def train(X, y, depth, width, residual, steps=2000, lr=0.01, seed=0):
    rng = np.random.default_rng(seed)
    Ws0, bs0 = init_net(depth, width, seed)
    Ws = [W.copy() for W in Ws0]
    bs = [b.copy() for b in bs0]
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
    pred = np.tanh(forward(X, Ws, bs, residual)[-1] @ w)
    loss = np.mean((pred - y) ** 2)
    dW0 = np.linalg.norm(Ws[0] - Ws0[0])      # bottom-layer weight change
    dWtop = np.linalg.norm(Ws[-1] - Ws0[-1])  # top-layer weight change
    return loss, dW0, dWtop

rng = np.random.default_rng(3)
X = rng.normal(0, 1, (128, 16))
y = np.sign(np.sin(X[:, 0] * X[:, 1]) + X[:, 2] ** 2 - X[:, 3])   # structured nonlinear target
y = (y > 0).astype(float)

print("\n=== Demo 2: structured target, depth=20, 2000 steps ===")
for residual in (False, True):
    loss, dW0, dWtop = train(X, y, depth=20, width=16, residual=residual)
    print(f"residual={residual}: loss={loss:.4f} | bottom ||dW0||={dW0:.2e} | top ||dWtop||={dWtop:.2e}")
