import numpy as np
import matplotlib.pyplot as plt

N = 50
a, b = 1.0, -0.01
xbar = 1.0
gammas = [1e-3, 1e-2, 1e-1, 1.0]

i = np.arange(1, N+1)[:, None]      
k = np.arange(1, N+1)[None, :]      
G = np.tril((a**(i-k)) * b)         # G[i,k] = a^(i-k) b for i>=k else 0
h = (a**np.arange(1, N+1)).reshape(-1, 1) * xbar  

U_list, X_list = [], []

for gamma in gammas:
    A = np.vstack([G, np.sqrt(gamma)*np.eye(N)])
    rhs = np.vstack([-h, np.zeros((N,1))])
    u, *_ = np.linalg.lstsq(A, rhs, rcond=None)   # least square by lstsq
    x = h + G @ u
    U_list.append(u.ravel());  X_list.append(x.ravel())

# controls
plt.figure()
for u, g in zip(U_list, gammas):
    plt.plot(np.arange(1, N+1), u, label=rf'$\gamma={g}$')
plt.xlabel('i'); plt.ylabel('u_i'); plt.grid(True)
plt.legend(); plt.title('Optimal controls vs $\gamma$')
plt.tight_layout()
plt.savefig('IIa_controls.png', dpi=150)

# states
plt.figure()
for x, g in zip(X_list, gammas):
    plt.plot(np.arange(0, N+1), np.r_[xbar, x], label=rf'$\gamma={g}$')
plt.xlabel('i'); plt.ylabel('x_i'); plt.grid(True)
plt.legend(); plt.title('State trajectories vs $\gamma$')
plt.tight_layout()
plt.savefig('IIa_states.png', dpi=150)

