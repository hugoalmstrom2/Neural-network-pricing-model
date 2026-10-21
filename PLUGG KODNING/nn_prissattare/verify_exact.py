"""
Kontroll av Black-Scholes-formeln innan den används som facit.

1. Symboliskt med sympy: PDE, slutvillkor, put-call-paritet och grekformlerna
2. Monte Carlo
3. Crank-Nicolson (numerisk lösning av PDE:n) ska konvergera mot formeln med ordning 2
"""
import json
import os

import numpy as np
import sympy as sp
from scipy.stats import chi2

from black_scholes import call_price, crank_nicolson_call, monte_carlo_call

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultat")
os.makedirs(OUT, exist_ok=True)
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(s)


def is_zero(expr):
    return sp.simplify(sp.expand(sp.powsimp(sp.expand(expr), combine="exp"))) == 0


# 1. Symboliskt ------------------------------------------------------------------
K, sig, tau = sp.symbols("K sigma tau", positive=True)
y, r = sp.symbols("y r", real=True)          # y = ln(S/K)
S = K * sp.exp(y)
Phi = lambda z: (1 + sp.erf(z / sp.sqrt(2))) / 2
phi = lambda z: sp.exp(-z**2 / 2) / sp.sqrt(2 * sp.pi)

d1 = (y + (r + sig**2 / 2) * tau) / (sig * sp.sqrt(tau))
d2 = d1 - sig * sp.sqrt(tau)
C = S * Phi(d1) - K * sp.exp(-r * tau) * Phi(d2)
P = K * sp.exp(-r * tau) * Phi(-d2) - S * Phi(-d1)

dS = lambda f: sp.diff(f, y) / S             # d/dS = (1/S) d/dy
C_S, C_tau = dS(C), sp.diff(C, tau)
C_SS = dS(C_S)

checks = {
    "S phi(d1) = K e^(-r tau) phi(d2)": S * phi(d1) - K * sp.exp(-r * tau) * phi(d2),
    "PDE": -C_tau + sig**2 * S**2 * C_SS / 2 + r * S * C_S - r * C,
    "put-call-paritet": C - P - (S - K * sp.exp(-r * tau)),
    "delta = N(d1)": C_S - Phi(d1),
    "gamma = phi(d1)/(S sigma sqrt(tau))": C_SS - phi(d1) / (S * sig * sp.sqrt(tau)),
    "vega = S phi(d1) sqrt(tau)": sp.diff(C, sig) - S * phi(d1) * sp.sqrt(tau),
    "dC/dtau": C_tau - (S * phi(d1) * sig / (2 * sp.sqrt(tau)) + r * K * sp.exp(-r * tau) * Phi(d2)),
    "rho = K tau e^(-r tau) N(d2)": sp.diff(C, r) - K * tau * sp.exp(-r * tau) * Phi(d2),
}
log("1. Symboliskt (sympy), uttrycket ska förenklas till 0")
symbolic = []
for name, expr in checks.items():
    ok = bool(is_zero(expr))
    symbolic.append({"name": name, "ok": ok})
    log(f"   {name:<38} {'ok' if ok else 'FEL'}")

# Slutvillkoret: när tau -> 0+ går d1, d2 mot +oo om S > K och -oo om S < K, så C -> max(S - K, 0)
u = sp.symbols("u", positive=True)
lims = [sp.limit(dd.subs(y, s * u), tau, 0, "+") for dd in (d1, d2) for s in (1, -1)]
ok = lims == [sp.oo, -sp.oo, sp.oo, -sp.oo]
symbolic.append({"name": "slutvillkor C -> max(S - K, 0)", "ok": bool(ok)})
log(f"   {'slutvillkor C -> max(S - K, 0)':<38} {'ok' if ok else 'FEL'}")

f_sym = sp.lambdify((y, tau, r, sig), C.subs(K, 1), "mpmath")
pts = [(0.8, 0.5, 0.02, 0.2), (1.2, 1.5, 0.07, 0.45), (1.0, 0.1, 0.0, 0.1)]
diff_impl = max(abs(float(f_sym(np.log(m_), t_, r_, s_)) - call_price(m_, t_, r_, s_)) for m_, t_, r_, s_ in pts)
log(f"   numpy-koden mot sympy-formeln: {diff_impl:.1e}")

# 2. Monte Carlo -----------------------------------------------------------------
log()
log("2. Monte Carlo, 4 000 000 banor per punkt")
rng = np.random.default_rng(2024)
mc_points = [
    (0.70, 0.25, 0.01, 0.15), (0.85, 1.00, 0.03, 0.30), (1.00, 0.50, 0.05, 0.20),
    (1.00, 2.00, 0.00, 0.45), (1.15, 0.10, 0.08, 0.10), (1.30, 1.50, 0.04, 0.35),
    (0.95, 0.75, 0.06, 0.50), (1.40, 2.00, 0.08, 0.25), (0.60, 0.10, 0.00, 0.10),
    (0.65, 0.50, 0.02, 0.12),
]
log(f"   {'m':>5} {'tau':>5} {'r':>5} {'sigma':>5}  {'formel':>12} {'MC':>12} {'std.fel':>9} {'z':>6}")
zs = []
for m_, t_, r_, s_ in mc_points:
    exact = call_price(m_, t_, r_, s_)
    est, se = monte_carlo_call(m_, t_, r_, s_, 4_000_000, rng)
    zs.append((est - exact) / se)
    log(f"   {m_:5.2f} {t_:5.2f} {r_:5.2f} {s_:5.2f}  {exact:12.6e} {est:12.6e} {se:9.2e} {zs[-1]:6.2f}")
zs = np.array(zs)
chi2_stat = float(np.sum(zs**2))
p_value = float(chi2.sf(chi2_stat, len(zs)))
log(f"   chi2 = {chi2_stat:.2f} med {len(zs)} frihetsgrader, p = {p_value:.2f}")

# 3. Crank-Nicolson ----------------------------------------------------------------
log()
log("3. Crank-Nicolson, max-fel mot formeln för 0.6 <= m <= 1.4")
m_eval = np.linspace(0.6, 1.4, 81)
cn = {}
for t_, r_, s_ in [(1.0, 0.03, 0.25), (0.25, 0.08, 0.10), (2.0, 0.0, 0.50)]:
    exact = call_price(m_eval, t_, r_, s_)
    key = f"tau={t_}, r={r_}, sigma={s_}"
    cn[key] = []
    log(f"   {key}")
    prev = None
    for n_x in (200, 400, 800, 1600, 3200):
        err = float(np.max(np.abs(crank_nicolson_call(m_eval, t_, r_, s_, n_x=n_x, n_t=n_x // 2) - exact)))
        cn[key].append({"n_x": n_x, "err": err})
        log(f"     n_x = {n_x:4d}: {err:.3e}" + (f"   ordning {np.log2(prev / err):.2f}" if prev else ""))
        prev = err

with open(os.path.join(OUT, "verifiering_exakt.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join(OUT, "sammanfattning_exakt.json"), "w", encoding="utf-8") as f:
    json.dump({"symbolic": symbolic, "impl_diff": float(diff_impl),
               "mc": {"n_points": len(zs), "max_abs_z": float(np.abs(zs).max()), "chi2": chi2_stat,
                      "p_value": p_value, "paths": 4_000_000},
               "cn": cn}, f, indent=2, ensure_ascii=False)
