import numpy as np

# ========== 网络与正反向（全部裸 NumPy，20 个隐藏层）==========
def init_net(depth, width, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, 0.1, (width, width)) for _ in range(depth)], \
           [np.zeros(width) for _ in range(depth)]

def relu(h):
    return np.maximum(h, 0.0)

def forward(x, Ws, bs, residual):
    """hs[0]=输入, hs[l+1]=第 l 层输出。residual=True: a^l = ReLU(pre) + a^{l-1}（恒等捷径 +x）。"""
    hs = [x]; h = x
    for l, (W, b) in enumerate(zip(Ws, bs)):
        h = relu(h @ W.T + b)
        if residual:
            h = h + hs[-1]              # 跳连：输出 = 变换 + 输入
        hs.append(h)
    return hs

def grads(hs, Ws, residual):
    """返回 (激活梯度范数, 权重梯度Frobenius范数)，均按层序 l=0..L-1。loss = sum(a^L)。"""
    L = len(Ws)
    delta = np.ones_like(hs[-1])        # dL/da^L
    act_norms, w_norms = [], []
    for l in range(L - 1, -1, -1):
        a_prev, a_out = hs[l], hs[l + 1]
        pre_act = a_out - a_prev if residual else a_out
        der = (pre_act > 0).astype(float)        # ReLU 导数
        delta_pre = delta * der                  # dL/d pre^l
        w_norms.append(np.linalg.norm(delta_pre.T @ a_prev))   # ||dL/dW^l||_F
        g_prev = delta_pre @ Ws[l]               # dL/d a^{l-1}
        if residual:
            g_prev = g_prev + delta              # 恒等项 +1：跳连的灵魂
        act_norms.append(np.linalg.norm(g_prev))
        delta = g_prev
    return act_norms[::-1], w_norms[::-1]

np.set_printoptions(precision=4, suppress=True)
depth, width = 20, 20
x = np.random.default_rng(1).normal(0, 1, (1, width))    # 单样本，形状 (1,20)

# ---- Demo 1: 梯度流 —— 激活梯度范数 vs 层号（对照教材图 9.3）----
print("=== Demo 1: 梯度流 (loss=sum(a^L), 深度 20) ===")
for residual in (False, True):
    Ws, bs = init_net(depth, width, seed=0)
    hs = forward(x, Ws, bs, residual)
    act, wg = grads(hs, Ws, residual)
    sel = [0, 4, 8, 12, 16, 19]
    print(f"residual={residual}:")
    print("  层号         :", sel)
    print("  激活梯度范数 :", [f"{act[i]:.3e}" for i in sel])

# ---- Demo 2: 一步学习 —— 底层权重在一个 SGD 步里动多少 ----
lr = 0.01
print("\n=== Demo 2: 一步 SGD 后底层权重的移动量 (lr=0.01) ===")
for residual in (False, True):
    Ws, bs = init_net(depth, width, seed=0)
    hs = forward(x, Ws, bs, residual)
    act, wg = grads(hs, Ws, residual)
    dW0, dWtop = lr * wg[0], lr * wg[-1]
    print(f"residual={residual}: ||ΔW_底层||={dW0:.3e} | ||ΔW_顶层||={dWtop:.3e} | 1000 步后底层累计 ≈ {1000*dW0:.3e}")
