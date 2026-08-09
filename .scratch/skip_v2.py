import numpy as np

def init_net(depth, width, seed=0):
    rng = np.random.default_rng(seed)
    Ws, bs = [], []
    for _ in range(depth):
        Ws.append(rng.normal(0, 0.1, (width, width)))
        bs.append(np.zeros(width))
    return Ws, bs

def forward(x, Ws, bs, residual):
    hs = [x]
    h = x
    for l, (W, b) in enumerate(zip(Ws, bs)):
        pre = h @ W.T + b
        h = np.tanh(pre)
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
        tanh_der = 1.0 - np.tanh(pre_act) ** 2
        delta_pre = delta * tanh_der
        g_prev = delta_pre @ Ws[l]
        if residual:
            g_prev = g_prev + delta
        norms.append(np.linalg.norm(g_prev))
        delta = g_prev
    return norms[::-1]

np.set_printoptions(precision=4, suppress=True)

# ===== Demo 1: gradient flow vs depth (Figure 9.3) =====
depth, width = 30, 20
x = np.random.default_rng(1).normal(0, 1, width)
print("=== Demo 1: gradient norm reaching each layer ===")
for residual in (False, True):
    Ws, bs = init_net(depth, width, seed=0)
    hs = forward(x, Ws, bs, residual)
    norms = layer_grad_norms(hs, Ws, residual)
    selected = [0, 5, 10, 15, 20, 25, 29]
    print(f"residual={residual}:")
    print("  layer        :", selected)
    print("  grad norm    :", np.round([norms[i] for i in selected], 4))

# ===== Demo 2: trainability (deep plain vs residual) =====
def train(X, y, depth, width, residual, steps=600, lr=0.03, seed=0):
    Ws, bs = init_net(depth, width, seed)
    for t in range(steps):
        hs = forward(X, Ws, bs, residual)
        hL = hs[-1]
        pred = np.tanh(hL)              # scalar output
        dL = 2.0 * (pred - y) * (1 - pred ** 2)
        delta = dL
        for l in range(depth - 1, -1, -1):
            a_prev = hs[l]
            a_out = hs[l + 1]
            pre_act = a_out - a_prev if residual else a_out
            tanh_der = 1.0 - np.tanh(pre_act) ** 2
            delta_pre = delta * tanh_der
            gW = delta_pre[:, None] * a_prev[None, :]
            Ws[l] = Ws[l] - lr * gW
            bs[l] = bs[l] - lr * delta_pre.mean(axis=0)
            g_prev = delta_pre @ Ws[l]
            if residual:
                g_prev = g_prev + delta
            delta = g_prev
    return Ws, bs

rng = np.random.default_rng(3)
X = rng.normal(0, 1, (64, 20))
y = (rng.random(64) > 0.5).astype(float)   # random labels -> memorization task

print("\n=== Demo 2: fit random labels, 15 hidden layers, 600 steps ===")
for residual in (False, True):
    Ws, bs = train(X, y, depth=15, width=20, residual=residual)
    hs = forward(X, Ws, bs, residual)
    pred = np.tanh(hs[-1])
    acc = np.mean((pred > 0) == y)
    print(f"residual={residual}: final accuracy = {acc:.3f}")
