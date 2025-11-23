import numpy as np
import matplotlib.pyplot as plt

train = np.loadtxt("trainingIb.dat")
val   = np.loadtxt("validationIa.dat")

x,  y,  yp = train[:,0], train[:,1], train[:,2]
xv, yv     = val[:,0],   val[:,1]

def Phi(x, n):
    x = np.asarray(x).ravel()
    return np.vander(x, N=n, increasing=True)

def Phi_dx(x, n):
    x = np.asarray(x).ravel()
    D = np.zeros((x.size, n))
    for j in range(1, n):
        D[:, j] = j * x**(j-1)
    return D

def mse(a, b):
    return np.mean((a-b)**2)

Nmax = 20
mse_noGrad   = np.zeros(Nmax)
mse_withGrad = np.zeros(Nmax)

for n in range(1, Nmax+1):

    A0 = Phi(x, n)
    theta0, *_ = np.linalg.lstsq(A0, y, rcond=None)
    mse_noGrad[n-1] = mse(Phi(xv, n) @ theta0, yv)

    A0 = Phi(x, n)
    A1 = Phi_dx(x, n)
    A_aug = np.vstack((A0, A1))
    b_aug = np.hstack((y, yp))
    theta_g, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    mse_withGrad[n-1] = mse(Phi(xv, n) @ theta_g, yv)

ns = np.arange(1, Nmax+1)

plt.figure(figsize=(6,4))
plt.semilogy(ns, mse_noGrad,  'o-', label='no derivative')
plt.semilogy(ns, mse_withGrad,'s-', label='with derivative')
plt.axhline(1e-3, linestyle='--', color='gray')
plt.xlabel('Polynomial degree n')
plt.ylabel('Validation MSE (log scale)')
plt.title('MSE vs n')
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig('q1b_MSE_vs_n.png', dpi=150)

best_n = np.argmin(mse_withGrad) + 1
print(best_n)
print(np.min(mse_withGrad))

theta_no, *_ = np.linalg.lstsq(Phi(x, best_n), y, rcond=None)

A_aug = np.vstack((Phi(x, best_n), Phi_dx(x, best_n)))
b_aug = np.hstack((y, yp))
theta_g, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)

xx = np.linspace(min(x.min(), xv.min()), max(x.max(), xv.max()), 400)
yy_no = Phi(xx, best_n) @ theta_no
yy_g  = Phi(xx, best_n) @ theta_g

plt.figure(figsize=(7,4))
plt.scatter(x,  y,  s=25, c='gray',   label='training data')
plt.scatter(xv, yv, s=25, c='orange', marker='x', label='validation points')
plt.plot(xx, yy_no, 'b',  label='no derivative fit')
plt.plot(xx, yy_g , 'r--',label='with derivative fit')
plt.xlabel('x')
plt.ylabel('V(x)')
plt.title(f'fitted curves (n = {best_n})')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('q1b_fitted_curves.png', dpi=150)

M = len(x)
ms = np.arange(5, M+1, 3)

mse_m_noGrad   = []
mse_m_withGrad = []

for m in ms:
    xm  = x[:m]
    ym  = y[:m]
    ypm = yp[:m]
    
    theta0, *_ = np.linalg.lstsq(Phi(xm, best_n), ym, rcond=None)
    mse_m_noGrad.append(mse(Phi(xv, best_n) @ theta0, yv))

    A0 = Phi(xm, best_n)
    A1 = Phi_dx(xm, best_n)
    A_aug = np.vstack((A0, A1))
    b_aug = np.hstack((ym, ypm))
    theta_g, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    mse_m_withGrad.append(mse(Phi(xv, best_n) @ theta_g, yv))

mse_m_noGrad   = np.array(mse_m_noGrad)
mse_m_withGrad = np.array(mse_m_withGrad)

plt.figure(figsize=(6,4))
plt.semilogy(ms, mse_m_noGrad,   'o-', label='no derivative')
plt.semilogy(ms, mse_m_withGrad, 's--', label='with derivative')
plt.xlabel('Number of training samples m')
plt.ylabel('Validation MSE (log scale)')
plt.title(f'MSE vs m (using n = {best_n})')
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("q1b_MSE_vs_m.png", dpi=150)
