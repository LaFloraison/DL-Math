import numpy as np

# ========== 网络与正反向 ==========
def init_net(depth, width, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, 0.1, (width, width)) for _ in range(depth)], \
           [np.zeros(width) for _ in range(depth)]

def relu(h):
    return np.maximum(h, 0.0)

def forward(x, Ws, bs, residual):
    """hs[0]=输入x, hs[l+1]=第l层输出。residual=True 时: a^l = ReLU(pre) + a^{l-1}。"""
    hs = [x]; h = x
    for l, (W, b) in enumerate(zip(Ws, bs)):
        h = relu(h @ W.T + b)
        if residual:
            h = h + hs[-1]          # 恒等捷径: +x
        hs.append(h)
    return hs

def grads(hs, Ws, residual):
    """返回 (激活梯度范数列表, 权重梯度Frobenius范数列表)，均按层序 l=0..L-1。"""
    L = len(Ws)
    delta = np.ones_like(hs[-1])                 # dL/da^L, loss = sum(a^L)
    act_norms, w_norms = [], []
    for l in range(L - 1, -1, -1):
        a_prev = hs[l]; a_out = hs[l + 1]
        pre_act = a_out - a_prev if residual else a_out
        der = (pre_act > 0).astype(float)        # ReLU 导数
        delta_pre = delta * der                  # dL/d pre^l
        gW = delta_pre.T @ a_prev                # dL/d W^l
        w_norms.append(np.linalg.norm(gW))       # Frobenius 范数
        g_prev = delta_pre @ Ws[l]               # dL/d a^{l-1}
        if residual:
            g_prev = g_prev + delta              # 恒等项 +1 (跳连的关键)
        act_norms.append(np.linalg.norm(g_prev))
        delta = g_prev
    return act_norms[::-1], w_norms[::-1]

np.set_printoptions(precision=4, suppress=True)
depth, width = 20, 20
x = np.random.default_rng(1).normal(0, 1, (1, width))   # batch of 1, shape (1, width)

print("=== Demo 1: 梯度流 —— 激活梯度范数 vs 层号 (对照教材图9.3) ===")
for residual in (False, True):
    Ws, bs = init_net(depth, width, seed=0)
    hs = forward(x, Ws, bs, residual)
    act, wg = grads(hs, Ws, residual)
    sel = [0, 4, 8, 12, 16, 19]
    print(f"residual={residual}:")
    print("  层号        :", sel)
    print("  激活梯度范数:", [f"{act[i]:.3e}" for i in sel])
    print("  权重梯度范数:", [f"{wg[i]:.3e}" for i in sel])

print("\n=== Demo 2: 一步学习 —— 底层权重在一步 SGD 后移动多少 ===")
lr = 0.01
for residual in (False, True):
    Ws, bs = init_net(depth, width, seed=0)
    hs = forward(x, Ws, bs, residual)
    act, wg = grads(hs, Ws, residual)
    dW0 = lr * wg[0]           # 底层(第1层)一步的权重移动量级
    dWtop = lr * wg[-1]        # 顶层(最后一层)一步的权重移动量级
    print(f"residual={residual}: ||ΔW_底层|| = lr*g0 = {dW0:.3e} | ||ΔW_顶层|| = lr*g_top = {dWtop:.3e}")
