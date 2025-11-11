import numpy as np
import matplotlib.pyplot as plt

# 读取数据：trainingIb.dat 三列 (x, V, V')
#            validationIa.dat 两列 (x, V)
train = np.loadtxt('trainingIb.dat')
val   = np.loadtxt('validationIa.dat')

x,  y,  yp  = train[:,0], train[:,1], train[:,2]
xv, yv      = val[:,0],   val[:,1]

def vander_inc(x, n):
    x = np.asarray(x).ravel()
    return np.vander(x, N=n, increasing=True)

def vander_inc_d1(x, n):
    x = np.asarray(x).ravel()
    if n == 1:
        return np.zeros((x.size, 1))
    cols = np.arange(n)           # 0,1,...,n-1
    dcols = np.hstack(([0], cols[1:]))  # 导数系数
    X = np.vander(x, N=n, increasing=True)
    X[:,1:] *= cols[1:]           # 对 x^j 乘 j
    return X

def mse(yhat, y):
    return float(np.mean((np.ravel(yhat)-np.ravel(y))**2))

Nmax = 20
MSE_noGrad  = np.empty(Nmax)
MSE_withGrad= np.empty(Nmax)
w = 1.0

for n in range(1, Nmax+1):
    # 仅函数值
    A = vander_inc(x, n)
    theta, *_ = np.linalg.lstsq(A, y, rcond=None)
    MSE_noGrad[n-1] = mse(vander_inc(xv, n) @ theta, yv)

    # 函数 + 导数
    A0 = vander_inc(x, n)
    A1 = vander_inc_d1(x, n)
    A_aug = np.vstack((A0, w*A1))
    b_aug = np.hstack((y,  w*yp))
    theta_g, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    MSE_withGrad[n-1] = mse(vander_inc(xv, n) @ theta_g, yv)

plt.figure()
plt.plot(np.arange(1,Nmax+1), MSE_noGrad, 'o-', label='without gradient')
plt.plot(np.arange(1,Nmax+1), MSE_withGrad, 's-', label='with gradient')
plt.xlabel('Degree n'); plt.ylabel('Validation MSE')
plt.title('Gradient-enhanced polynomial regression')
plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig('fig_Ib_compare.png', dpi=150)

best_n = np.argmin(MSE_withGrad) + 1
A_noGrad = vander_inc(x, best_n)
theta_noGrad, *_ = np.linalg.lstsq(A_noGrad, y, rcond=None)

A_grad = np.vstack((A_noGrad, w*vander_inc_d1(x, best_n)))
b_grad = np.hstack((y, w*yp))
theta_grad, *_ = np.linalg.lstsq(A_grad, b_grad, rcond=None)

xx = np.linspace(min(x.min(), xv.min()), max(x.max(), xv.max()), 400)
yy_noGrad = vander_inc(xx, best_n) @ theta_noGrad
yy_grad   = vander_inc(xx, best_n) @ theta_grad

plt.figure()
plt.scatter(x, y, s=20, c='gray', label='training points')
plt.plot(xx, yy_noGrad, 'b', label='without gradient')
plt.plot(xx, yy_grad, 'r--', label='with gradient')
plt.legend(); plt.grid(True)
plt.title(f'Fitted curves comparison (n={best_n})')
plt.tight_layout()
plt.savefig('fig_Ib_fit_compare.png', dpi=150)