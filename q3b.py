import numpy as np
import matplotlib.pyplot as plt

N = 50
a, b, xbar = 1.0, 0.05, 1.0
c, d, ybar = 0.2, -0.5, 1.0

def build_G(alpha, beta, N):
    i = np.arange(1, N+1)[:, None]
    k = np.arange(1, N+1)[None, :]
    return np.tril((alpha**(i-k)) * beta)      

def build_h(alpha, N, x0):
    return (alpha**np.arange(1, N+1)).reshape(-1,1) * x0

Gx, hx = build_G(a, b, N), build_h(a, N, xbar)
Gy, hy = build_G(c, d, N), build_h(c, N, ybar)

glist = 10.0**np.arange(-5, 5.0 + 1e-12, 0.1)
J1 = np.empty_like(glist)
J2 = np.empty_like(glist)
Jtot = np.empty_like(glist)

for t, gamma in enumerate(glist):
    A = np.block([
        [Gx,                       np.zeros((N, N))],
        [np.zeros((N, N)),         Gy            ],
        [np.sqrt(gamma)*np.eye(N), -np.sqrt(gamma)*np.eye(N)]
    ])
    rhs = np.vstack([-hx, -hy, np.zeros((N,1))])
    w, *_ = np.linalg.lstsq(A, rhs, rcond=None)   # [u; v]
    u = w[:N]; v = w[N:]
    x = hx + Gx @ u
    y = hy + Gy @ v
    J1[t] = float(np.sum(x**2) + np.sum(y**2))
    J2[t] = float(np.sum((u - v)**2))
    Jtot[t] = J1[t] + float(gamma) * J2[t]

plt.figure()
plt.semilogx(glist, J1, label=r'$J_1(\gamma)=\|x^*\|^2+\|y^*\|^2$')
plt.semilogx(glist, J2, label=r'$J_2(\gamma)=\|u^*-v^*\|^2$')
plt.grid(True, which='both', axis='x')
plt.xlabel(r'$\gamma$'); plt.ylabel('value')
plt.title('II.c — J1(γ) and J2(γ)')
plt.legend(); plt.tight_layout()
plt.savefig('IIc_J1_J2_vs_gamma.png', dpi=150)

plt.figure()
plt.plot(J1, J2, '-', lw=1)
plt.xlabel(r'$J_1$'); plt.ylabel(r'$J_2$'); plt.grid(True)
plt.title('II.c — Pareto front (J1 vs J2)')
plt.tight_layout()
plt.savefig('IIc_Pareto_front.png', dpi=150)


