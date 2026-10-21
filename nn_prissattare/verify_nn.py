"""
Black-Scholes-nätet mot den exakta lösningen.

  python verify_nn.py [--model resultat/modell.pt] [--tag standard]

1. fel på 10^6 testpunkter   2. greker   3. PDE-residual   4. arbitragevillkor
5. felgräns från maximumprincipen (se README)
"""
import argparse
import json
import os

import numpy as np

from black_scholes import DOMAIN, bs_pde_residual, call_greeks, call_price, sample_domain
from nn_model import load_model, nn_price, nn_price_and_derivatives
from plot_style import SEQ_CMAP, SERIES, apply_style, plt
from train import make_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "resultat")
ORE = 1e4   # c -> öre per option med K = 100 kr

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(OUT, "modell.pt"))
ap.add_argument("--tag", default="standard")
args = ap.parse_args()

model, ckpt = load_model(args.model)
cfg = ckpt["config"]
lines, summary = [], {"config": cfg}


def log(s=""):
    print(s, flush=True)
    lines.append(s)


log(f"{os.path.relpath(args.model, HERE)}: {cfg['depth']} x {cfg['width']}, {cfg['n_params']} parametrar, {cfg['epochs']} epoker, "
    f"sobolev={cfg['sobolev']}, wide={cfg.get('wide', False)}, hard={cfg.get('hard', 0.0)}")

# 1. Fel ------------------------------------------------------------------------
N = 1_000_000
X = sample_domain(N, np.random.default_rng(777))
exact = call_price(*X.T)
e = nn_price(model, X) - exact
ae = np.abs(e)
mask = exact >= 0.01
rel = ae[mask] / exact[mask]
Xtr, ytr, _ = make_dataset(cfg["n_train"], cfg["seed"], cfg.get("wide", False), cfg.get("hard", 0.0))
etr = np.abs(nn_price(model, Xtr) - ytr)

err = {"MAE": float(ae.mean()), "RMSE": float(np.sqrt(np.mean(e**2))), "max": float(ae.max()),
       "p99.9": float(np.quantile(ae, 0.999)), "bias": float(e.mean()),
       "R2": float(1 - np.sum(e**2) / np.sum((exact - exact.mean()) ** 2)),
       "rel_mean": float(rel.mean()), "rel_max": float(rel.max()),
       "train_MAE": float(etr.mean()), "train_max": float(etr.max())}
summary["errors"] = err
w = X[np.argmax(ae)]
log()
log(f"1. Fel på {N} testpunkter (i öre per option, K = 100 kr)")
log(f"   MAE {err['MAE'] * ORE:.2f}   RMSE {err['RMSE'] * ORE:.2f}   99,9 % {err['p99.9'] * ORE:.2f}   "
    f"max {err['max'] * ORE:.2f}   bias {err['bias'] * ORE:+.3f}")
log(f"   relativt fel där C >= 1 kr: medel {rel.mean():.2e}, max {rel.max():.2e}")
log(f"   sämsta punkt: m={w[0]:.3f}, tau={w[1]:.3f}, r={w[2]:.4f}, sigma={w[3]:.3f}")
log(f"   träningsdata: MAE {etr.mean() * ORE:.2f}, max {etr.max() * ORE:.2f}  (test/träning {ae.mean() / etr.mean():.3f})")
log(f"   Inga av {N} punkter har större fel än {ae.max() * ORE:.1f} öre. Med 95 % konfidens är andelen")
log(f"   av domänen med större fel då under {-np.log(0.05) / N:.1e}, eftersom (1-p)^N <= 0,05.")

# 2. Greker ---------------------------------------------------------------------
Xg = X[:200_000]
mg, tg, rg, sg = Xg.T
d = nn_price_and_derivatives(model, Xg)
g = call_greeks(*Xg.T)
summary["greeks"] = {}
log()
log("2. Greker från nätet (autograd) mot formlerna, 200 000 punkter")
log(f"   {'':8} {'MAE':>9} {'max':>9} {'rel. MAE':>9}")
for k in ["delta", "gamma", "vega", "dtau", "rho"]:
    ek = np.abs(d[k] - g[k])
    s = {"MAE": float(ek.mean()), "max": float(ek.max()), "rel_MAE": float(ek.mean() / np.mean(np.abs(g[k])))}
    summary["greeks"][k] = s
    log(f"   {k:8} {s['MAE']:9.2e} {s['max']:9.2e} {s['rel_MAE']:9.2%}")

# 3. PDE ------------------------------------------------------------------------
R_nn = bs_pde_residual(d["c"], d["delta"], d["gamma"], d["dtau"], mg, rg, sg)
R_ex = bs_pde_residual(call_price(*Xg.T), g["delta"], g["gamma"], g["dtau"], mg, rg, sg)
scale = np.mean(np.abs(g["dtau"]))
summary["pde"] = {"mean": float(np.mean(np.abs(R_nn))), "p99": float(np.quantile(np.abs(R_nn), 0.99)),
                  "max": float(np.max(np.abs(R_nn))), "rel": float(np.mean(np.abs(R_nn)) / scale)}
log()
log("3. PDE-residual  R = -c_tau + 1/2 sigma^2 m^2 c_mm + r m c_m - r c")
log(f"   formeln: max |R| = {np.max(np.abs(R_ex)):.1e}")
log(f"   nätet:   medel |R| = {summary['pde']['mean']:.2e}, max {summary['pde']['max']:.2e}, "
    f"{summary['pde']['rel']:.2%} av medel |c_tau|")

# 4. Arbitragevillkor -------------------------------------------------------------
c_nn = d["c"]
price_err = np.abs(c_nn - call_price(*Xg.T)).max()
gr = summary["greeks"]
conds = {
    "c >= max(m - e^(-r tau), 0)": (np.maximum(mg - np.exp(-rg * tg), 0) - c_nn, price_err),
    "c <= m": (c_nn - mg, price_err),
    "delta >= 0": (-d["delta"], gr["delta"]["max"]),
    "delta <= 1": (d["delta"] - 1, gr["delta"]["max"]),
    "gamma >= 0": (-d["gamma"], gr["gamma"]["max"]),
    "vega >= 0": (-d["vega"], gr["vega"]["max"]),
    "dc/dtau >= 0": (-d["dtau"], gr["dtau"]["max"]),
    "rho >= 0": (-d["rho"], gr["rho"]["max"]),
}
summary["arbitrage"] = {}
log()
log("4. Arbitragevillkor: andel punkter med brott, största brott, nätets max-fel i samma storhet")
for name, (viol, ref) in conds.items():
    frac, worst = float(np.mean(viol > 0)), float(max(viol.max(), 0.0))
    summary["arbitrage"][name] = {"frac": frac, "max": worst, "ref_err": float(ref)}
    log(f"   {name:<28} {frac:8.3%} {worst:10.2e} {ref:10.2e}")

# 5. Maximumprincipen -------------------------------------------------------------
log()
log("5. Felgräns |e| <= max_rand |e| + (tau - tau_min) max |R|, för 15 fixa (r, sigma)")
log(f"   {'r':>5} {'sigma':>5} {'randfel':>9} {'max|R|':>9} {'max|e| inne':>11} {'gräns':>9}")
m_grid = np.linspace(*DOMAIN["m"], 241)
t_grid = np.linspace(*DOMAIN["tau"], 241)
M, T = np.meshgrid(m_grid, t_grid, indexing="ij")
rows, all_ok = [], True
for r_ in (0.0, 0.04, 0.08):
    for s_ in (0.1, 0.2, 0.3, 0.4, 0.5):
        Xs = np.column_stack([M.ravel(), T.ravel(), np.full(M.size, r_), np.full(M.size, s_)])
        ds = nn_price_and_derivatives(model, Xs)
        es = (ds["c"] - call_price(*Xs.T)).reshape(M.shape)
        Rs = bs_pde_residual(ds["c"], ds["delta"], ds["gamma"], ds["dtau"], Xs[:, 0], r_, s_)
        b0 = max(np.abs(es[:, 0]).max(), np.abs(es[0, :]).max(), np.abs(es[-1, :]).max())
        rmax = np.abs(Rs).max()
        bound = b0 + (T - DOMAIN["tau"][0]) * rmax
        ok = bool(np.all(np.abs(es) <= bound + 1e-12))
        all_ok &= ok
        inner = np.abs(es[1:-1, 1:])
        rows.append({"r": r_, "sigma": s_, "boundary": float(b0), "Rmax": float(rmax),
                     "interior_max": float(inner.max()), "bound_max": float(bound.max()), "ok": ok})
        log(f"   {r_:5.2f} {s_:5.2f} {b0:9.2e} {rmax:9.2e} {inner.max():11.2e} {bound.max():9.2e} {'' if ok else 'BRYTS'}")
summary["max_principle"] = {"all_ok": all_ok, "rows": rows}
log(f"   gränsen håller i alla snitt: {'ja' if all_ok else 'nej'}")

with open(os.path.join(OUT, f"verifiering_nn_{args.tag}.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join(OUT, f"sammanfattning_{args.tag}.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

# Figurer -------------------------------------------------------------------------
apply_style()
r0, s0 = 0.03, 0.25
m_line = np.linspace(*DOMAIN["m"], 401)
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
for t_, col in zip([0.25, 1.0, 2.0], SERIES):
    Xl = np.column_stack([m_line, np.full_like(m_line, t_), np.full_like(m_line, r0), np.full_like(m_line, s0)])
    nl = nn_price_and_derivatives(model, Xl)
    gl = call_greeks(*Xl.T)
    gl["c"] = call_price(*Xl.T)
    for ax, key in zip(axes.flat[:3], ["c", "delta", "gamma"]):
        ax.plot(m_line, gl[key], color=col, lw=1, label=f"exakt, τ = {t_}")
        ax.plot(m_line[::25], nl[key][::25], "o", color=col, mfc="none", label=f"nät, τ = {t_}")
    axes[1, 1].plot(m_line, (nl["c"] - gl["c"]) * ORE, color=col, label=f"τ = {t_}")
for ax, title in zip(axes.flat, ["pris c = C/K", "delta", "gamma (K d²C/dS²)", "fel nät - exakt (öre)"]):
    ax.set_title(title)
    ax.set_xlabel("S/K")
axes[0, 0].legend(fontsize=8)
axes[1, 1].legend(fontsize=8)
fig.suptitle(f"Nät mot exakt lösning, r = {r0}, σ = {s0}")
fig.tight_layout()
fig.savefig(os.path.join(OUT, f"fig_snitt_{args.tag}.png"))
plt.close(fig)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
nb = 24
mi = np.minimum(((X[:, 0] - DOMAIN["m"][0]) / np.ptp(DOMAIN["m"]) * nb).astype(int), nb - 1)
ti = np.minimum(((X[:, 1] - DOMAIN["tau"][0]) / np.ptp(DOMAIN["tau"]) * nb).astype(int), nb - 1)
H = np.zeros((nb, nb))
np.maximum.at(H, (ti, mi), ae)
im = ax1.imshow(H * ORE, origin="lower", aspect="auto", cmap=SEQ_CMAP, extent=[*DOMAIN["m"], *DOMAIN["tau"]])
fig.colorbar(im, ax=ax1, label="största fel i cellen (öre)")
ax1.grid(False)
ax1.set_xlabel("S/K")
ax1.set_ylabel("τ (år)")
ax1.set_title("Största fel per cell (alla r och σ)")
ax2.hist(e * ORE, bins=120)
ax2.set_yscale("log")
ax2.set_xlabel("fel (öre)")
ax2.set_title("Felfördelning")
fig.tight_layout()
fig.savefig(os.path.join(OUT, f"fig_fel_{args.tag}.png"))
plt.close(fig)

h = ckpt["history"]
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(h["epoch"], np.sqrt(h["train_mse"]), label="träning, RMSE")
ax.semilogy(h["epoch"], h["val_rmse"], label="validering, RMSE")
ax.semilogy(h["epoch"], h["val_max"], label="validering, max")
ax.set_xlabel("epok")
ax.set_ylabel("fel i c")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, f"fig_traning_{args.tag}.png"))
plt.close(fig)
