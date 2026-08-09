import numpy as np                      # 唯一依赖：NumPy

rng = np.random.default_rng(42)         # 固定随机种子，结果可复现

def propagate(Ws, x, use_relu=False):
    """让输入 x 依次穿过每一层权重，返回每层输出的标准差"""
    stds = []
    for i, W in enumerate(Ws):
        x = x @ W.T                     # 前向传播：y = x @ W^T
        if use_relu and i < len(Ws) - 1:
            x = np.maximum(x, 0.0)      # 除最后一层外，加一道 ReLU
        stds.append(np.std(x))
    return np.array(stds)

def naive_init(n_in, n_out):
    """朴素初始化：标准正态 σ=1，权重太大"""
    return rng.normal(0.0, 1.0, (n_out, n_in))

def small_init(n_in, n_out):
    """朴素初始化：均匀小值 ±0.05，权重太小"""
    return rng.uniform(-0.05, 0.05, (n_out, n_in))

def xavier_init(n_in, n_out):
    """Xavier：均匀分布，边界 a = sqrt(6 / (n_in + n_out))"""
    a = np.sqrt(6.0 / (n_in + n_out))
    return rng.uniform(-a, a, (n_out, n_in))

def he_init(n_in, n_out):
    """He：正态分布，标准差 σ = sqrt(2 / n_in)"""
    return rng.normal(0.0, np.sqrt(2.0 / n_in), (n_out, n_in))

L = 50                                  # 网络深度：50 层
n = 256                                 # 每层 256 个神经元
x0 = rng.normal(0.0, 1.0, (64, n))      # 输入：64 个样本、256 维，σ=1

# ---- ① 纯线性链：不同初始化下，信号穿过 50 层后还剩多大 ----
for name, init in [("naive(σ=1)", naive_init),
                   ("small(±0.05)", small_init),
                   ("xavier", xavier_init)]:
    stds = propagate([init(n, n) for _ in range(L)], x0.copy())
    print(f"{name}: 首层 σ={stds[0]:.3f}   第50层 σ={stds[-1]:.3e}")

# ---- ② 加 ReLU：Xavier 会消失，He 能稳住 ----
for name, init in [("xavier", xavier_init), ("he", he_init)]:
    stds = propagate([init(n, n) for _ in range(L)], x0.copy(), use_relu=True)
    print(f"{name}: 首层 σ={stds[0]:.3f}   第50层 σ={stds[-1]:.3e}")

# ---- ③ 单层验证：n_in=4 的 Xavier，对 500 次随机初始化取平均 ----
mean_var = 0.0
for _ in range(500):
    W4 = xavier_init(4, 4)
    x4 = rng.normal(0.0, 1.0, (1000, 4))
    mean_var += ((x4 @ W4.T) ** 2).mean()
print(f"n_in=4: a={np.sqrt(6/8):.4f}   500 次初始化的平均 E[Var(y)] = {mean_var/500:.4f}（理论值 1）")
