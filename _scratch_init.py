import numpy as np                      # 唯一依赖：NumPy

rng = np.random.default_rng(42)         # 固定种子，结果可复现

def propagate(Ws, x, use_relu=False):
    """依次穿过每一层权重，返回每层输出的标准差序列"""
    stds = []
    for i, W in enumerate(Ws):
        x = x @ W.T                     # 前向：y = x @ W^T（线性变换）
        if use_relu and i < len(Ws) - 1:
            x = np.maximum(x, 0.0)      # 除最后一层外，加一道 ReLU
        stds.append(np.std(x))
    return np.array(stds)

def naive_init(n_in, n_out):
    return rng.normal(0.0, 1.0, (n_out, n_in))          # sigma_w = 1：太大

def small_init(n_in, n_out):
    return rng.uniform(-0.05, 0.05, (n_out, n_in))      # 均匀小值：太小

def xavier_init(n_in, n_out):
    a = np.sqrt(6.0 / (n_in + n_out))                   # Xavier 边界
    return rng.uniform(-a, a, (n_out, n_in))

def he_init(n_in, n_out):
    return rng.normal(0.0, np.sqrt(2.0 / n_in), (n_out, n_in))  # He：sigma=sqrt(2/n_in)

L = 50                                  # 50 层深链
n = 256                                 # 每层 256 个神经元
x0 = rng.normal(0.0, 1.0, (64, n))      # 输入：64 个样本、256 维，sigma=1

# ---- 1 纯线性链：看朴素初始化如何放大/缩小信号 ----
for name, init in [("naive(sigma=1)", naive_init),
                   ("small(+/-0.05)", small_init),
                   ("xavier", xavier_init)]:
    stds = propagate([init(n, n) for _ in range(L)], x0.copy())
    print(f"[linear] {name:14s} layer1 sigma={stds[0]:6.3f}  layer50 sigma={stds[-1]:.3e}")

# ---- 2 加 ReLU：Xavier 失效，He 稳住 ----
for name, init in [("xavier", xavier_init), ("he", he_init)]:
    stds = propagate([init(n, n) for _ in range(L)], x0.copy(), use_relu=True)
    print(f"[relu]   {name:14s} layer1 sigma={stds[0]:6.3f}  layer50 sigma={stds[-1]:.3e}")

# ---- 3 单层方差验证：n_in=4，对 500 次随机初始化的方差取平均 ----
sum_var = 0.0
for _ in range(500):
    W4 = xavier_init(4, 4)
    x4 = rng.normal(0.0, 1.0, (1000, 4))
    sum_var += ((x4 @ W4.T)**2).mean()
print(f"n_in=4 xavier: a={np.sqrt(6/8):.4f}  E[Var(y)] over 500 draws = {sum_var/500:.4f}  (theory = 1)")
