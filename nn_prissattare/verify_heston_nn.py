"""
Heston-nätet mot COS-referensen.

  python verify_heston_nn.py                                                  # IV-nätet
  python verify_heston_nn.py --model resultat/heston_modell_400.pt --tag 400  # pris-nätet

Fel i pris och implicit volatilitet, delta, Heston-PDE:ns residual (autograd),
arbitragevillkor, volatilitetsleende och tidsåtgång per option.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from black_scholes import implied_vol
from heston import DOMAIN, FEATURES, cos_call, mc_call, pde_residual, quad_call, sample
from nn_model import load_model, nn_derivatives, nn_price
from plot_style import SEQ_CMAP, SERIES, apply_style, plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "resultat")
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(OUT, "heston_iv.pt"))
ap.add_argument("--tag", default="iv", help="läggs till i filnamnen, t.ex. 400")
args = ap.parse_args()
sfx = f"_{args.tag}" if args.tag else ""

model, ckpt = load_model(args.model)
cfg = ckpt["config"]
lines, res = [], {"config": cfg}


def log(s=""):
    print(s, flush=True)
    lines.append(s)


log(f"Modell: {cfg['depth']} x {cfg['width']}, {cfg['n_params']} parametrar, {cfg['epochs']} epoker, "
    f"sobolev={cfg['sobolev']}")

# --- 1. Prisfel -------------------------------------------------------------
N = 100_000
X = sample(N, np.random.default_rng(777))
t0 = time.time()
ref, ref_delta = cos_call(*X.T, with_delta=True)
t_cos = (time.time() - t0) / N
pred = nn_price(model, X)
e = pred - ref
ae = np.abs(e)
mask = ref >= 0.01
rel = ae[mask] / ref[mask]

res["errors"] = {"MAE": float(ae.mean()), "RMSE": float(np.sqrt(np.mean(e**2))), "max": float(ae.max()),
                 "p99.9": float(np.quantile(ae, 0.999)), "rel_mean": float(rel.mean())}
log()
log(f"1. Prisfel på {N} testpunkter (c = C/K; x 10 000 = öre vid K = 100 kr)")
log(f"   MAE {ae.mean():.2e}  RMSE {np.sqrt(np.mean(e**2)):.2e}  99,9 % {np.quantile(ae, 0.999):.2e}  max {ae.max():.2e}")
log(f"   relativt fel där C >= 1 % av K: medel {rel.mean():.2e}")
# Träningsdatan återskapas av train_heston.py; saknas den hoppas jämförelsen med träningsfelet över
data_path = os.path.join(OUT, "heston_data.npz")
if os.path.exists(data_path):
    d = np.load(data_path)
    e_train = np.abs(nn_price(model, d["X"]) - d["y"])
    res["errors"]["train_MAE"] = float(e_train.mean())
    log(f"   tränings-MAE {e_train.mean():.2e} (test/träning = {ae.mean() / e_train.mean():.3f})")
else:
    log("   (träningsdata saknas, kör train_heston.py för att jämföra med träningsfelet)")
w = X[np.argmax(ae)]
log("   sämsta punkt: " + ", ".join(f"{f}={v:.3f}" for f, v in zip(FEATURES, w)))

# --- 2. Implicit volatilitet --------------------------------------------------
# Bara optioner med tidsvärde >= 0,5 % av K. Är tidsvärdet litet (djupt ITM eller OTM, kort
# löptid) är vega nästan noll och ett litet prisfel blir ett stort fel i implicit vol.
time_value = ref - np.maximum(X[:, 0] - np.exp(-X[:, 2] * X[:, 1]), 0)
sel = (time_value >= 0.005) & (np.abs(np.log(X[:, 0])) <= 0.3)
iv_ref = implied_vol(ref[sel], X[sel, 0], X[sel, 1], X[sel, 2])
iv_nn = implied_vol(pred[sel], X[sel, 0], X[sel, 1], X[sel, 2])
div = 100 * (iv_nn - iv_ref)
ok = np.isfinite(div)
res["iv"] = {"n": int(ok.sum()), "MAE": float(np.mean(np.abs(div[ok]))),
             "p99": float(np.quantile(np.abs(div[ok]), 0.99)), "max": float(np.max(np.abs(div[ok]))),
             "missing": int((~ok).sum())}
log()
log(f"2. Fel i implicit volatilitet (volpunkter), {ok.sum()} optioner med tidsvärde >= 0,5 % av K och |ln m| <= 0,3")
log(f"   MAE {res['iv']['MAE']:.3f}  99 % {res['iv']['p99']:.3f}  max {res['iv']['max']:.3f}"
    f"  (saknas för {res['iv']['missing']} st)")

# --- 3. Delta, PDE och arbitragevillkor (autograd) ----------------------------
n_g = 20_000
Xg = X[:n_g]
c, g, h = nn_derivatives(model, Xg, pairs=((0, 0), (3, 3), (0, 3)))
d_err = np.abs(g[:, 0] - ref_delta[:n_g])
m_, tau_, r_, v_, k_, th_, xi_, rho_ = Xg.T
R = pde_residual(c, g[:, 0], h[(0, 0)], g[:, 3], h[(3, 3)], h[(0, 3)], g[:, 1], m_, r_, v_, k_, th_, xi_, rho_)
scale = np.mean(np.abs(g[:, 1]))
res["delta"] = {"MAE": float(d_err.mean()), "max": float(d_err.max())}
res["pde"] = {"mean": float(np.mean(np.abs(R))), "p99": float(np.quantile(np.abs(R), 0.99)),
              "max": float(np.max(np.abs(R))), "rel": float(np.mean(np.abs(R)) / scale)}
log()
log(f"3. Delta: MAE {d_err.mean():.2e}, max {d_err.max():.2e}")
log(f"   Heston-PDE: medel |R| {res['pde']['mean']:.2e}, 99 % {res['pde']['p99']:.2e}, max {res['pde']['max']:.2e}")
log(f"   relativt termernas storlek (medel |c_tau| = {scale:.3f}): {res['pde']['rel']:.2%}")

conds = {
    "c >= max(m - e^-r tau, 0)": np.maximum(m_ - np.exp(-r_ * tau_), 0) - c,
    "c <= m": c - m_,
    "0 <= delta <= 1": np.maximum(-g[:, 0], g[:, 0] - 1),
    "gamma >= 0": -h[(0, 0)],
    "dc/dtau >= 0": -g[:, 1],
    "dc/dv0 >= 0": -g[:, 3],
}
res["arbitrage"] = {}
log("   arbitragevillkor (andel punkter med brott, största brott):")
for name, viol in conds.items():
    res["arbitrage"][name] = {"frac": float(np.mean(viol > 0)), "max": float(max(viol.max(), 0))}
    log(f"     {name:<26} {np.mean(viol > 0):8.3%}  {max(viol.max(), 0):.1e}")

# --- 4. Tidsåtgång ------------------------------------------------------------
m32 = load_model(args.model)[0].float()
Xt = torch.tensor(X, dtype=torch.float32)
with torch.no_grad():
    m32(Xt)
    t0 = time.time()
    for _ in range(5):
        m32(Xt)
t_nn = (time.time() - t0) / (5 * N)
t0 = time.time()
cos_call(*X[:20_000].T, N=128, L=(12, 12))
t_cos_fast = (time.time() - t0) / 20_000
t0 = time.time()
for x in X[:50]:
    quad_call(*x)
t_quad = (time.time() - t0) / 50
t0 = time.time()
mc_call(*X[0], n_paths=100_000, n_steps=250, rng=np.random.default_rng(0))
t_mc = time.time() - t0
res["timing"] = {"nn": t_nn, "cos_ref": t_cos, "cos_fast": t_cos_fast, "quad": t_quad, "mc": t_mc}
log()
log("4. Tid per option (CPU)")
log(f"   nätet (batch om {N})      {t_nn * 1e6:9.2f} mikrosek")
log(f"   COS, N=128 (snabb)        {t_cos_fast * 1e6:9.2f} mikrosek")
log(f"   COS, N=1024 (referens)    {t_cos * 1e6:9.2f} mikrosek")
log(f"   quad (Gil-Pelaez)         {t_quad * 1e6:9.0f} mikrosek")
log(f"   Monte Carlo (100k banor)  {t_mc * 1e6:9.0f} mikrosek")
log(f"   Nätet är {t_cos / t_nn:.0f} gånger snabbare än referens-COS och {t_mc / t_nn:.0e} gånger snabbare än MC.")

with open(os.path.join(OUT, f"heston_verifiering{sfx}.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
with open(os.path.join(OUT, f"heston_sammanfattning{sfx}.json"), "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)

# --- Figurer --------------------------------------------------------------------
apply_style()
cases = [
    ("typisk aktie", dict(tau=0.5, r=0.02, v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)),
    ("hög vol-of-vol", dict(tau=0.25, r=0.02, v0=0.09, kappa=1.0, theta=0.06, xi=0.9, rho=-0.9)),
    ("lång löptid", dict(tau=2.0, r=0.05, v0=0.02, kappa=3.0, theta=0.05, xi=0.3, rho=-0.3)),
]
strikes = np.linspace(0.75, 1.3, 56)                 # K/S
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
for ax, (name, p) in zip(axes, cases):
    m = 1 / strikes
    Xs = np.column_stack([m] + [np.full_like(m, p[f]) for f in FEATURES[1:]])
    iv_c = implied_vol(cos_call(*Xs.T), m, p["tau"], p["r"])
    iv_n = implied_vol(nn_price(model, Xs), m, p["tau"], p["r"])
    ax.plot(strikes, 100 * iv_c, color=SERIES[1], label="COS (referens)")
    ax.plot(strikes[::4], 100 * iv_n[::4], "o", color=SERIES[0], mfc="none", mew=1.4, ms=6, label="nätet")
    ax.set_title(f"{name}, τ = {p['tau']}")
    ax.set_xlabel("K / S")
axes[0].set_ylabel("implicit volatilitet (%)")
axes[0].legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, f"fig_heston_leende{sfx}.png"))
plt.close(fig)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
nb = 20
xi_i = np.minimum(((X[:, 6] - DOMAIN["xi"][0]) / np.ptp(DOMAIN["xi"]) * nb).astype(int), nb - 1)
ti = np.minimum(((X[:, 1] - DOMAIN["tau"][0]) / np.ptp(DOMAIN["tau"]) * nb).astype(int), nb - 1)
H = np.zeros((nb, nb))
np.maximum.at(H, (ti, xi_i), ae)
im = ax1.imshow(H * 1e4, origin="lower", aspect="auto", cmap=SEQ_CMAP, extent=[*DOMAIN["xi"], *DOMAIN["tau"]])
cb = fig.colorbar(im, ax=ax1)
cb.set_label("största fel i cellen (öre)")
cb.outline.set_visible(False)
ax1.grid(False)
ax1.set_xlabel("vol of vol ξ")
ax1.set_ylabel("τ (år)")
ax1.set_title("Var felet är störst")
ax2.hist(div[ok], bins=100, color=SERIES[0])
ax2.set_yscale("log")
ax2.set_xlabel("fel i implicit volatilitet (volpunkter)")
ax2.set_title("Fel i implicit vol")
fig.tight_layout()
fig.savefig(os.path.join(OUT, f"fig_heston_fel{sfx}.png"))
plt.close(fig)
log("\nFigurer sparade i resultat/")
