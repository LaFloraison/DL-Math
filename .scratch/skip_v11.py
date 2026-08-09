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

# Diagnose hidden norm growth
rng = np.random.default_rng(0)
x = rng.normal(0, 1, 16)
print("hL norm by depth and scale:")
for scale in (0.1, 0.03, 0.02):
    for depth in (10, 15, 20):
        Ws, bs = init_net(depth, 16, scale, seed=0)
        hs = forward(x, Ws, bs, True)
        hL = hs[-1]
        print(f"  scale={scale} depth={depth}: ||hL||={np.linalg.norm(hL):.2f}  ||x||={np.linalg.norm(x):.2f}")
