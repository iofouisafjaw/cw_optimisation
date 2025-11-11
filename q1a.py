import numpy as np
import matplotlib.pyplot as plt

# ---------- 读取数据 ----------
train = np.loadtxt('trainingIa.dat')
val   = np.loadtxt('validationIa.dat')
x_tr, y_tr = train[:, 0], train[:, 1]
x_va, y_va = val[:, 0],   val[:, 1]

# ---------- 设计矩阵 ----------
def vander_inc(x, n):
    """升幂 Vandermonde 矩阵: [1, x, x^2, ..., x^(n-1)]"""
    return np.vander(np.asarray(x).ravel(), N=n, increasing=True)

def mse(yhat, y):
    return float(np.mean((np.ravel(yhat) - np.ravel(y))**2))

# ---------- 1) MSE vs n ----------
Nmax = 20
MSEn = np.empty(Nmax)
for n in range(1, Nmax + 1):
    A = vander_inc(x_tr, n)
    theta, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    yhat_va = vander_inc(x_va, n) @ theta
    MSEn[n - 1] = mse(yhat_va, y_va)

# 找到第一个使 MSE <= 1e-3 的 n
target = 1e-3
idx = np.where(MSEn <= target)[0]
best_n = int(idx[0] + 1) if idx.size else 10
print(f'Best n (first <=1e-3): {best_n}, MSE={MSEn[best_n-1]:.3e}')

# ---------- 对数坐标绘图 ----------
plt.figure()
plt.semilogy(np.arange(1, Nmax + 1), MSEn, '-o', label='Validation MSE')
plt.axhline(target, ls='--', color='r', label='threshold 1e-3')
plt.scatter([best_n], [MSEn[best_n-1]], s=80, c='k', zorder=5)
plt.annotate(f'n={best_n}\nMSE={MSEn[best_n-1]:.2e}',
             xy=(best_n, MSEn[best_n-1]),
             xytext=(best_n + 0.6, MSEn[best_n-1]*2),
             arrowprops=dict(arrowstyle='->', lw=1))

plt.grid(True, which='both', axis='y')
plt.xlabel('Degree n')
plt.ylabel('Validation MSE (log scale)')
plt.title('Part I.a — MSE vs degree n (log scale)')
plt.legend()
plt.tight_layout()
plt.savefig('fig_mse_vs_n_log.png', dpi=150)

# ---------- 2) 固定 best_n：MSE vs 训练点数 ----------
n = best_n
m_list = np.arange(5, len(x_tr) + 1)
MSEm = np.empty_like(m_list, dtype=float)
for k, m in enumerate(m_list):
    A = vander_inc(x_tr[:m], n)
    theta, *_ = np.linalg.lstsq(A, y_tr[:m], rcond=None)
    yhat_va = vander_inc(x_va, n) @ theta
    MSEm[k] = mse(yhat_va, y_va)

plt.figure()
plt.plot(m_list, MSEm, '-o')
plt.grid(True)
plt.xlabel('# training points')
plt.ylabel('Validation MSE')
plt.title(f'Part I.a — MSE vs #training points (n={n})')
plt.tight_layout()
plt.savefig('fig_mse_vs_m.png', dpi=150)

# ---------- 3) 拟合曲线 ----------
A = vander_inc(x_tr, n)
theta, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
xx = np.linspace(min(x_tr.min(), x_va.min()),
                 max(x_tr.max(), x_va.max()), 400)
yy = vander_inc(xx, n) @ theta

plt.figure()
plt.scatter(x_tr, y_tr, s=20, c='b', label='train')
plt.scatter(x_va, y_va, s=20, c='r', label='validation')
plt.plot(xx, yy, 'k', lw=1.5, label='fit')
plt.grid(True)
plt.legend()
plt.title(f'Part I.a — fitted polynomial (n={n})')
plt.tight_layout()
plt.savefig('fig_fit.png', dpi=150)


