import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-2, 2, 400)
abs_z = np.abs(z)

def L_eps(z, eps):
    z = np.asarray(z)
    out = np.where(np.abs(z) <= eps, 0.5/eps * z**2,
                   np.abs(z) - 0.5*eps)
    return out

plt.figure(figsize=(6,4))
plt.plot(z, abs_z, 'k-', label='|z|')
plt.plot(z, L_eps(z, 1.0), 'r--', label='L_eps, eps=1')
plt.plot(z, L_eps(z, 0.1), 'b-.', label='L_eps, eps=0.1')
plt.xlabel('z'); plt.ylabel('value')
plt.title('Smooth approximation of |z|')
plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig('L_eps_vs_abs.png', dpi=150)
plt.close()
