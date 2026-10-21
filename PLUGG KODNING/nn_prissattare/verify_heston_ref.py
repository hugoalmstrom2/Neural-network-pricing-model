"""
Kontroll av Heston-referensen (COS) innan den används som facit för nätet.

1. Publicerat testvärde (Fang & Oosterlee 2008)
2. COS mot quad (två olika Fourier-metoder) på slumpade punkter och domänens hörn
3. COS-felet mot antal termer N
4. xi -> 0 ska ge Black-Scholes med effektiv volatilitet
5. Monte Carlo
6. PDE-residualen för COS-priset (finita differenser)
"""
import itertools
import json
import os

import numpy as np

from black_scholes import call_price
from heston import DOMAIN, cos_call, mc_call, pde_residual, quad_call, sample

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultat")
os.makedirs(OUT, exist_ok=True)
lines, res = [], {}


def log(s=""):
    print(s, flush=True)
    lines.append(s)


# 1. Testfall från Fang & Oosterlee (2008): S = K = 100, T = 1, r = 0 -> 5.785155450
p_fo = (1.0, 1.0, 0.0, 0.0175, 1.5768, 0.0398, 0.5751, -0.5711)
c_cos, c_quad = 100 * cos_call(*p_fo)[0], 100 * quad_call(*p_fo)
log("1. Publicerat testvärde 5.785155450")
log(f"   COS  {c_cos:.9f}   (diff {c_cos - 5.785155450:+.1e})")
log(f"   quad {c_quad:.9f}   (diff {c_quad - 5.785155450:+.1e})")
res["benchmark"] = {"published": 5.785155450, "cos": c_cos, "quad": c_quad}

# 2. COS mot quad
rng = np.random.default_rng(5)
X = sample(300, rng)
corners = [[m, t, 0.0, v, k, th, xi, -0.9] for m, t, v, k, th, xi in
           itertools.product(DOMAIN["m"], DOMAIN["tau"], DOMAIN["v0"], DOMAIN["kappa"], DOMAIN["theta"], DOMAIN["xi"])]
X = np.vstack([X, corners])
q = np.array([quad_call(*x) for x in X])
e = np.abs(cos_call(*X.T) - q)
log()
log(f"2. COS mot quad, {len(X)} punkter (300 slumpade + {len(corners)} hörn)")
log(f"   max {e.max():.1e}   99%-percentil {np.quantile(e, 0.99):.1e}   median {np.median(e):.1e}")
res["cos_vs_quad"] = {"n": len(X), "max": float(e.max()), "p99": float(np.quantile(e, 0.99)), "median": float(np.median(e))}

# 3. Konvergens i N för några vanliga punkter
log()
log("3. COS-fel mot antal termer N (max över 20 punkter, symmetriskt intervall L=12)")
Xc, qc = X[:20], q[:20]
conv = []
for N in (16, 32, 64, 128, 256):
    err = np.abs(cos_call(*Xc.T, N=N, L=(12, 12)) - qc).max()
    conv.append({"N": N, "err": float(err)})
    log(f"   N={N:4d}  {err:.1e}")
res["cos_convergence"] = conv
log("   Felet faller ungefär exponentiellt i N tills trunkeringen av intervallet tar över.")

# 4. Gränsen xi -> 0: variansen blir deterministisk och priset ska bli Black-Scholes
#    med sigma_eff^2 = (1/tau) * int_0^tau v(t) dt
log()
log("4. xi -> 0 mot Black-Scholes med effektiv volatilitet")
m, tau, r, v0, kappa, theta = 0.95, 1.0, 0.03, 0.06, 2.0, 0.03
sig_eff = np.sqrt((theta * tau + (v0 - theta) * (1 - np.exp(-kappa * tau)) / kappa) / tau)
bs = call_price(m, tau, r, sig_eff)
res["xi_limit"] = {}
for rho in (-0.7, 0.0):
    prev, rows = None, []
    log(f"   rho = {rho}:")
    for xi in (0.2, 0.1, 0.05, 0.025, 0.0125):
        diff = abs(cos_call(m, tau, r, v0, kappa, theta, xi, rho)[0] - bs)
        order = np.log2(prev / diff) if prev else float("nan")
        rows.append({"xi": xi, "diff": float(diff), "order": float(order)})
        log(f"     xi={xi:<7} |Heston - BS| = {diff:.2e}   ordning {order:4.2f}")
        prev = diff
    res["xi_limit"][str(rho)] = rows
log("   Med rho != 0 är första korrektionen proportionell mot rho*xi (ordning 1),")
log("   med rho = 0 mot xi^2 (ordning 2). Skillnaden går mot 0 som den ska.")

# 5. Monte Carlo (Euler har en liten diskretiseringsbias, så detta är en rimlighetskontroll)
log()
log("5. Monte Carlo, 400 000 banor, 500 tidssteg per år")
mc_pts = [(1.0, 1.0, 0.03, 0.04, 2.0, 0.04, 0.5, -0.7), (0.8, 0.5, 0.0, 0.09, 1.0, 0.06, 0.8, -0.5),
          (1.2, 2.0, 0.05, 0.02, 3.0, 0.05, 0.3, -0.3), (0.9, 0.25, 0.02, 0.16, 4.0, 0.1, 0.9, -0.9),
          (1.1, 1.5, 0.07, 0.05, 0.8, 0.15, 0.6, 0.0)]
rng = np.random.default_rng(11)
zs = []
for p in mc_pts:
    est, se = mc_call(*p, n_paths=400_000, n_steps=int(500 * p[1]), rng=rng)
    ref = cos_call(*p)[0]
    zs.append((est - ref) / se)
    log(f"   {np.round(p, 3)}  COS {ref:.6f}  MC {est:.6f} +- {se:.1e}  z = {zs[-1]:+.2f}")
res["mc"] = {"z": [float(z) for z in zs], "bias_check": []}
# Om någon punkt avviker: kör om med olika steglängd. Beror avvikelsen på Euler-bias ska den krympa.
worst = mc_pts[int(np.argmax(np.abs(zs)))]
ref = cos_call(*worst)[0]
log(f"   Steglängdskontroll för {np.round(worst, 3)}:")
for spy in (125, 500, 2000, 8000):
    est, se = mc_call(*worst, n_paths=400_000, n_steps=int(spy * worst[1]), rng=np.random.default_rng(7))
    res["mc"]["bias_check"].append({"steps_per_year": spy, "diff": float(est - ref), "se": float(se)})
    log(f"     {spy:5d} steg/år: MC - COS = {est - ref:+.2e} (std.fel {se:.1e}, z = {(est - ref) / se:+.2f})")

# 6. PDE-residual för COS-priset med centrala differenser
log()
log("6. Heston-PDE:n för COS-priset (finita differenser, h = 1e-3)")
Xp = sample(200, np.random.default_rng(3))
h = 1e-3
f = lambda dm=0.0, dt=0.0, dv=0.0: cos_call(Xp[:, 0] + dm, Xp[:, 1] + dt, Xp[:, 2], Xp[:, 3] + dv, *Xp[:, 4:].T)
c0 = f()
c_m = (f(dm=h) - f(dm=-h)) / (2 * h)
c_mm = (f(dm=h) - 2 * c0 + f(dm=-h)) / h**2
c_v = (f(dv=h) - f(dv=-h)) / (2 * h)
c_vv = (f(dv=h) - 2 * c0 + f(dv=-h)) / h**2
c_mv = (f(dm=h, dv=h) - f(dm=h, dv=-h) - f(dm=-h, dv=h) + f(dm=-h, dv=-h)) / (4 * h * h)
c_tau = (f(dt=h) - f(dt=-h)) / (2 * h)
R = pde_residual(c0, c_m, c_mm, c_v, c_vv, c_mv, c_tau, Xp[:, 0], Xp[:, 2], Xp[:, 3], *Xp[:, 4:].T)
log(f"   medel |R| = {np.mean(np.abs(R)):.1e}, max |R| = {np.max(np.abs(R)):.1e} "
    f"(jämför medel |c_tau| = {np.mean(np.abs(c_tau)):.1e})")
res["pde_cos"] = {"mean": float(np.mean(np.abs(R))), "max": float(np.max(np.abs(R))),
                  "scale": float(np.mean(np.abs(c_tau)))}

with open(os.path.join(OUT, "heston_referens.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
with open(os.path.join(OUT, "heston_referens.json"), "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
