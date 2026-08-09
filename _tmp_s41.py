"""
§4.1 概率论导论 · 代码.md
从最离散的随机实验——掷骰子——出发，走完一条完整的路：
模拟随机采样 -> 计算经验频率 -> 对比理论概率 -> 可视化。

运行：python 代码.py    （绘图依赖 matplotlib）
"""

import random            # 标准库：伪随机数生成器（Mersenne Twister 算法）
import matplotlib.pyplot as plt

N_FACES = 6              # 骰子面数：样本空间 Omega = {1,2,3,4,5,6}
N_ROLLS = 1000           # 掷多少次骰子（采样量 N）
SEED = 42                # 随机种子：让"随机"实验可以精确复现


def roll_once(rng):
    """掷一次骰子：把一个均匀随机数映射到 1..6 的某一个面。"""
    u = rng.random()             # u ~ U[0,1)：抽一个 [0,1) 上的均匀随机数
    return int(u * N_FACES) + 1  # k = floor(6u) + 1 ∈ {1,...,6}


def simulate_rolls(n_rolls, seed=SEED):
    """独立地掷 n_rolls 次骰子，返回一个长度为 n_rolls 的结果列表。"""
    rng = random.Random(seed)    # 用固定种子初始化发生器，结果可复现
    return [roll_once(rng) for _ in range(n_rolls)]


def empirical_frequencies(rolls, n_faces=N_FACES):
    """把结果列表统计成每个面的经验频率 Phat(k) = 出现次数 / 总次数。"""
    counts = [0] * n_faces       # 6 个计数槽，下标 0..5 对应面 1..6
    for face in rolls:
        counts[face - 1] += 1    # 面 k 落在槽 k-1
    total = len(rolls)
    return [c / total for c in counts]


def theoretical_pmf(n_faces=N_FACES):
    """理论概率质量函数：公平骰子每个面 p(k) = 1/6，均匀离散分布。"""
    return [1.0 / n_faces] * n_faces


def plot_distribution(empirical, theoretical, labels=None):
    """并排对比经验频率（柱状）与理论概率（折线），观察两者是否靠拢。"""
    x = list(range(1, len(empirical) + 1))
    plt.bar(x, empirical, width=0.5, color="steelblue", alpha=0.8,
            label="经验频率 Phat(k)")
    plt.plot(x, theoretical, "ro-", label="理论概率 p(k) = 1/6")
    plt.xlabel("骰子面数 k")
    plt.ylabel("频率 / 概率")
    plt.xticks(x, labels)
    plt.ylim(0, 0.5)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.title("经验频率 vs 理论概率（N = %d 次掷骰）" % N_ROLLS)
    plt.show()


# ---------------- 主流程 ----------------
rolls = simulate_rolls(N_ROLLS)              # 1. 模拟采样
emp = empirical_frequencies(rolls)           # 2. 经验频率
th = theoretical_pmf()                       # 3. 理论概率

print("掷了 %d 次骰子，前 20 次结果为：" % N_ROLLS)
print("  ", rolls[:20])
print("经验频率 Phat(k) =", [round(f, 4) for f in emp])
print("理论概率 p(k)    =", [round(p, 4) for p in th])
plot_distribution(emp, th, labels=["1", "2", "3", "4", "5", "6"])

# 大数定律：样本量越大，经验频率越贴近理论概率（主线中"箭墙成形"）
print("\n大数定律（观察"面 6"的经验概率随样本量收敛）：")
for n in (10, 100, 1000, 10000, 100000):
    r = simulate_rolls(n)
    f6 = sum(1 for v in r if v == 6) / n
    print("  N = %6d 次 -> Phat(6) = %.4f   (理论值 1/6 = 0.1667)" % (n, f6))
