import numpy as np
import matplotlib.pyplot as plt

def simulate(a, b, u, x0):
    N = len(u)
    x = np.zeros(N + 1)
    x[0] = x0
    for i in range(N):
        x[i + 1] = a * x[i] + b * u[i]
    return x

def L_eps_prime(z, eps):
    z = np.asarray(z)
    return np.where(np.abs(z) <= eps,
                    z,
                    eps * np.sign(z))

def grad_J(u, v, params):
    a, b, c, d, x0, y0, gam1, gam2, eps = params
    N = len(u)

    x = simulate(a, b, u, x0)
    y = simulate(c, d, v, y0)

    du = np.zeros(N)
    dv = np.zeros(N)

    for i in range(N):
        du[i] += 2 * b * np.sum((a ** (np.arange(i, N) - i)) * x[i + 1:])
        dv[i] += 2 * d * np.sum((c ** (np.arange(i, N) - i)) * y[i + 1:])

        z = u[i] - v[i]
        l1p = L_eps_prime(z, eps)

        du[i] += 2 * gam2 * z + gam1 * l1p
        dv[i] += -2 * gam2 * z - gam1 * l1p

    return du, dv, x, y

def GD(
    N, a, b, c, d, x0, y0,
    gam1, gam2, eps,
    steps=5000, lr=1e-3, tol=1e-5
):
    u = np.zeros(N)
    v = np.zeros(N)
    params = (a, b, c, d, x0, y0, gam1, gam2, eps)

    for k in range(steps):
        du, dv, x, y = grad_J(u, v, params)
        gnorm = np.sqrt(np.sum(du * du) + np.sum(dv * dv))
        if gnorm < tol:
            break
        u -= lr * du
        v -= lr * dv

    x = simulate(a, b, u, x0)
    y = simulate(c, d, v, y0)
    return u, v, x, y

if __name__ == "__main__":
    N = 50
    a, b, x0 = 1.0, 0.05, 1.0
    c, d, y0 = 0.2, -0.5, 1.0

    cases = [
        (1.0, 1.0, 0.0, "case1"),
        (1.0, 0.0, 1.0, "case2"),
        (0.1, 0.0, 1.0, "case3"),
    ]

    for eps, gam2, gam1, name in cases:
        u, v, x, y = GD(
            N, a, b, c, d, x0, y0,
            gam1=gam1, gam2=gam2, eps=eps,
            steps=6000, lr=1e-3, tol=1e-5
        )

        # states
        plt.figure(figsize=(6, 4))
        plt.plot(range(N + 1), x, label="x")
        plt.plot(range(N + 1), y, label="y")
        plt.xlabel("time index")
        plt.ylabel("trajectory")
        plt.title(f"{name} (trajectory)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"q2d_{name}_trajectory.png", dpi=150)
        plt.close()

        # controls
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, N + 1), u, label="u")
        plt.plot(range(1, N + 1), v, label="v")
        plt.xlabel("time index")
        plt.ylabel("control")
        plt.title(f"{name} (controls)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"q2d_{name}_controls.png", dpi=150)
        plt.close()

