import numpy as np                       # NumPy：线性代数的瑞士军刀

# ---- ① 教材例：A = diag(1, 4, 0)，对角阵，奇异值一目了然 ----
A = np.array([[1., 0., 0.],
              [0., 4., 0.],
              [0., 0., 0.]])
print("A =\n", A)

# ---- ② 完整 SVD（默认 full_matrices=True）----
U, S, Vh = np.linalg.svd(A)
print("U:", U.shape, "| S:", S.shape, "| Vh:", Vh.shape)
print("U =\n", U)
print("奇异值 S =", S)
print("Vh =\n", Vh)

# ---- ③ 重构验证：A = U Σ Vᵀ，把一维 S 填进 m×n 对角块 ----
S_fill = np.zeros(A.shape)
np.fill_diagonal(S_fill, S)
recon = U @ S_fill @ Vh
print("重构最大误差:", np.abs(recon - A).max())

# ---- ④ 经济型 SVD：只算 min(m,n) 个，形状更小 ----
Ue, Se, Vhe = np.linalg.svd(A, full_matrices=False)
print("经济型:", Ue.shape, Se.shape, Vhe.shape)
print("经济型重构误差:", np.abs(Ue @ np.diag(Se) @ Vhe - A).max())

# ---- ⑤ 非方阵 + 截断：3×5 随机数据，保留 k 个奇异值 ----
rng = np.random.default_rng(0)
X = rng.standard_normal((3, 5))
U, S, Vh = np.linalg.svd(X, full_matrices=False)
print("X 形状:", X.shape, "| 经济型 U,S,Vh:", U.shape, S.shape, Vh.shape)
print("X 的奇异值 S =", np.round(S, 4))
for k in (1, 2, 3):
    Xk = U[:, :k] @ np.diag(S[:k]) @ Vh[:k, :]
    err = np.linalg.norm(X - Xk)                    # Frobenius 范数
    theory = np.sqrt(np.sum(S[k:]**2))              # Eckart-Young 预言
    print(f"k={k}: 实际误差 {err:.4f} | 理论误差 {theory:.4f} | 秩={np.linalg.matrix_rank(Xk)}")

# ---- ⑥ 奇异值谱：各方向携带的能量占比（前 r 个平方 / 总平方）----
print("能量占比:", np.round(S**2 / np.sum(S**2), 4))
print("累计能量占比:", np.round(np.cumsum(S**2) / np.sum(S**2), 4))
