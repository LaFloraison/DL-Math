import numpy as np
np.set_printoptions(precision=3, suppress=True)
def relu(h): return np.maximum(h, 0.0)
def init_net(depth, width, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, 0.1, (width, width)) for _ in range(depth)], [np.zeros(width) for _ in range(depth)]
depth, width = 20, 20
Ws, bs = init_net(depth, width, seed=0)
x = np.random.default_rng(1).normal(0, 1, (1, width))
hs = [x]; h = x
for W, b in zip(Ws, bs):
    h = relu(h @ W.T + b); hs.append(h)
# top layer l=19
a_prev = hs[19]; a_out = hs[20]
print("hs19 norm:", np.linalg.norm(a_prev), "hs20 norm:", np.linalg.norm(a_out))
delta = np.ones_like(a_out)
der = (a_out > 0).astype(float)
print("active units:", der.sum(), "delta_pre norm:", np.linalg.norm(delta*der))
gW = (delta*der).T @ a_prev
print("gW shape:", gW.shape, "F-norm:", np.linalg.norm(gW))
