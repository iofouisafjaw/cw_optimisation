import numpy as np
import matplotlib.pyplot as plt

N = 50
a, b, xbar = 1.0, 0.05, 1.0
c, d, ybar = 0.2, -0.5, 1.0
gamma = 1.0

def build_G(alpha, beta, N):
    i = np.arange(1, N+1)[:, None]
    k = np.arange(1, N+1)[None, :]
    return np.tril((alpha**(i-k)) * beta)      # G_{ik}=alpha^{i-k}*beta (i>=k)

def build_h(alpha, N, x0):
    return (alpha**np.arange(1, N+1)).reshape(-1,1) * x0

Gx, hx = build_G(a, b, N), build_h(a, N, xbar)
Gy, hy = build_G(c, d, N), build_h(c, N, ybar)

A = np.block([
    [Gx,                       np.zeros((N, N))],
    [np.zeros((N, N)),         Gy            ],
    [np.sqrt(gamma)*np.eye(N), -np.sqrt(gamma)*np.eye(N)]
])

rhs = np.vstack([-hx, -hy, np.zeros((N,1))])


w, *_ = np.linalg.lstsq(A, rhs, rcond=None)
u = w[:N].ravel()           
v = w[N:].ravel()           
x = (hx + Gx @ u.reshape(-1,1)).ravel()  
y = (hy + Gy @ v.reshape(-1,1)).ravel()   

plt.figure()
plt.plot(np.arange(0, N+1), np.r_[xbar, x], label='x*')
plt.plot(np.arange(0, N+1), np.r_[ybar, y], label='y*')
plt.xlabel('i'); plt.ylabel('state'); plt.grid(True)
plt.legend(); plt.title(f'II.b — state trajectories (γ={gamma:g})')
plt.tight_layout(); plt.savefig('IIb_states.png', dpi=150)

plt.figure()
plt.plot(np.arange(1, N+1), u, label='u*')
plt.plot(np.arange(1, N+1), v, label='v*')
plt.xlabel('i'); plt.ylabel('control'); plt.grid(True)
plt.legend(); plt.title(f'II.b — controls (γ={gamma:g})')
plt.tight_layout(); plt.savefig('IIb_controls.png', dpi=150)


