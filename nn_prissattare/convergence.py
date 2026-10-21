"""
Felet mot nätets storlek: tränar 4-lagersnät med olika bredd i 60 epoker var.

  python convergence.py              # tränar och ritar (ca 7 min)
  python convergence.py --plot-only  # ritar om figuren från resultat/konvergens.json
"""
import argparse
import json
import os

import numpy as np

from black_scholes import call_price, sample_domain
from nn_model import nn_price
from plot_style import apply_style, plt
from train import train

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultat")
PATH = os.path.join(OUT, "konvergens.json")

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--plot-only", action="store_true")
args = ap.parse_args()

if not args.plot_only:
    X = sample_domain(200_000, np.random.default_rng(777))
    exact = call_price(*X.T)
    rows = []
    print(f"{'bredd':>6} {'parametrar':>10} {'RMSE':>10} {'max':>10} {'tid (s)':>8}")
    for width in (4, 8, 16, 32, 64, 128):
        model, _, cfg = train(width=width, depth=4, epochs=args.epochs, verbose=False)
        e = nn_price(model, X) - exact
        rows.append({"width": width, "n_params": cfg["n_params"], "rmse": float(np.sqrt(np.mean(e**2))),
                     "max": float(np.abs(e).max()), "seconds": cfg["train_seconds"]})
        print(f"{width:6d} {cfg['n_params']:10d} {rows[-1]['rmse']:10.3e} {rows[-1]['max']:10.3e} "
              f"{cfg['train_seconds']:8.0f}", flush=True)
    slope = float(np.polyfit(np.log([r["n_params"] for r in rows]), np.log([r["rmse"] for r in rows]), 1)[0])
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump({"epochs": args.epochs, "rows": rows, "slope_rmse": slope}, f, indent=2)

with open(PATH, encoding="utf-8") as f:
    data = json.load(f)
rows = data["rows"]
P = np.array([r["n_params"] for r in rows])
apply_style()
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.loglog(P, [1e4 * r["rmse"] for r in rows], "o-", label="RMSE")
ax.loglog(P, [1e4 * r["max"] for r in rows], "o-", label="max")
for p, r in zip(P, rows):
    ax.annotate(str(r["width"]), (p, 1e4 * r["rmse"]), textcoords="offset points", xytext=(4, -12), fontsize=8)
ax.set_xlabel("antal parametrar")
ax.set_ylabel("fel (öre, K = 100 kr)")
ax.set_title(f"Fel mot nätstorlek, {data['epochs']} epoker (siffrorna är bredden)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_konvergens.png"))
print(f"lutning log(RMSE) mot log(parametrar): {data['slope_rmse']:.2f}")
