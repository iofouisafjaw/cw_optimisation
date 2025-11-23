import numpy as np
import matplotlib.pyplot as plt

train = np.loadtxt("trainingIa.dat")
val   = np.loadtxt("validationIa.dat")

x, y   = train[:,0], train[:,1]
xv, yv = val[:,0],   val[:,1]

def Phi(x, n):
    x = np.asarray(x).ravel()
    return np.vander(x, N=n, increasing=True)

def mse(a, b):
    return np.mean((a-b)**2)

Nmax = 20
mse_vs_n = np.zeros(Nmax)

for n in range(1, Nmax+1):
    A = Phi(x, n)
    theta, *_ = np.linalg.lstsq(A, y, rcond=None)
    mse_vs_n[n-1] = mse(Phi(xv, n) @ theta, yv)

plt.figure(figsize=(6,4))
plt.semilogy(range(1, Nmax+1), mse_vs_n, 'o-')
plt.axhline(1e-3, linestyle='--', color='gray')
plt.xlabel("Polynomial degree n")
plt.ylabel("Validation MSE")
plt.title("MSE vs n")
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("q1a_MSE_vs_n.png", dpi=150)

n_star = np.argmin(mse_vs_n) + 1

print(n_star)

A= Phi(x, n_star)
theta_star, *_ = np.linalg.lstsq(A, y, rcond=None)

xx = np.linspace(min(x.min(), xv.min()), max(x.max(), xv.max()), 400)
yy = Phi(xx, n_star) @ theta_star

plt.figure(figsize=(7,4))
plt.scatter(x, y, s=25, c='gray', label='training data')
plt.scatter(xv, yv, s=25, c='orange', marker='x', label='validation points')
plt.plot(xx, yy, 'r', label=f'poly fit (n={n_star})')
plt.xlabel("x")
plt.ylabel("V(x)")
plt.title(f"fitted curve (n = {n_star})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("q1a_fitted_curve.png", dpi=150)

M = len(x)
mse_vs_m = np.zeros(M)

for m in range(2, M+1):
    xm = x[:m]
    ym = y[:m]
    A_m = Phi(xm, n_star)
    theta_m, *_ = np.linalg.lstsq(A_m, ym, rcond=None)
    mse_vs_m[m-1] = mse(Phi(xv, n_star) @ theta_m, yv)

ms = np.arange(1, M+1)
plt.figure(figsize=(6,4))
plt.semilogy(ms, mse_vs_m, 'o-')
plt.xlabel("Number of training points m")
plt.ylabel("Validation MSE")
plt.title(f"MSE vs m (n = {n_star})")
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("q1a_MSE_vs_m.png", dpi=150)
