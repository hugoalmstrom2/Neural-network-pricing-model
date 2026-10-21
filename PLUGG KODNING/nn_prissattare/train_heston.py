"""
Träning av Heston-nätet. Indata (m, tau, r, v0, kappa, theta, xi, rho), utdata c = C/K.
Träningsdata räknas med COS och sparas i resultat/heston_data.npz första gången.

  python train_heston.py                  # 4 x 128, pris + delta, 200 epoker
  python train_heston.py --iv --epochs 400 --out resultat/heston_iv.pt

Med --iv lär sig nätet implicit volatilitet och priset räknas med Black-Scholes-formeln.
"""
import argparse
import os
import time

import numpy as np
import torch

from black_scholes import call_greeks, implied_vol
from heston import DOMAIN, FEATURES, WIDE_DOMAIN, cos_call
from nn_model import IVNet, PricerMLP, bs_call_torch
from train import fit, save

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "resultat")


def sample_points(n, rng, domain, hard=0.0):
    lo = np.array([domain[f][0] for f in FEATURES])
    hi = np.array([domain[f][1] for f in FEATURES])
    X = lo + (hi - lo) * rng.random((n, len(FEATURES)))
    k = int(hard * n)
    if k:
        # kort löptid, låg varians, nära ATM - samma typ av område som var svårast för BS-nätet
        X[:k, 0] = rng.uniform(0.85, 1.15, k)
        X[:k, 1] = rng.uniform(domain["tau"][0], 0.5, k)
        X[:k, 3] = rng.uniform(domain["v0"][0], 0.05, k)
    return X


def make_data(n, seed, domain, hard):
    X = sample_points(n, np.random.default_rng(seed), domain, hard)
    c, delta = cos_call(*X.T, with_delta=True)
    return X, c, delta


def load_or_make(n_train, seed, hard):
    path = os.path.join(RES, "heston_data.npz")
    if os.path.exists(path):
        d = np.load(path)
        if int(d["n_train"]) == n_train and int(d["seed"]) == seed and float(d["hard"]) == hard:
            print("Läser cachad data från resultat/heston_data.npz")
            return d["X"], d["y"], d["dy"], d["Xv"], d["yv"]
    t0 = time.time()
    X, y, dy = make_data(n_train, seed, WIDE_DOMAIN, hard)
    Xv, yv, _ = make_data(2**14, seed + 1000, DOMAIN, 0.0)
    print(f"COS-data för {n_train + 2**14} optioner: {time.time() - t0:.0f} s")
    np.savez(path, X=X, y=y, dy=dy, Xv=Xv, yv=yv, n_train=n_train, seed=seed, hard=hard)
    return X, y, dy, Xv, yv


def iv_labels(X, y, min_vega=1e-4):
    """
    Implicit vol för COS-priserna. Punkter där vega < min_vega tas bort: där ger COS-felet
    (ca 1e-9) för stort fel i volatiliteten, och priset är ändå nästan noll.
    """
    m, tau, r = X[:, 0], X[:, 1], X[:, 2]
    iv = implied_vol(y, m, tau, r)
    vega = call_greeks(m, tau, r, np.nan_to_num(iv, nan=0.2))["vega"]
    return iv, np.isfinite(iv) & (vega > min_vega)


def fit_iv(model, X, iv, dy, Xv, yv, epochs=400, batch=4096, lr=2e-3, sobolev=1.0, verbose=True):
    """Förlust: (sigma_nät - sigma)^2 + sobolev * (dc/dm - delta)^2, där c = BS(m, tau, r, sigma_nät)."""
    X_t = torch.tensor(X, dtype=torch.float32)
    iv_t = torch.tensor(iv, dtype=torch.float32)
    d_t = torch.tensor(dy, dtype=torch.float32)
    Xv_t = torch.tensor(Xv, dtype=torch.float32)
    n = len(X)
    steps = n // batch
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps, eta_min=1e-6)

    hist = {"epoch": [], "train_mse": [], "val_rmse": [], "val_max": []}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        running = 0.0
        for i in range(steps):
            idx = perm[i * batch:(i + 1) * batch]
            xb = X_t[idx].requires_grad_(sobolev > 0)
            sig = model.iv(xb)
            loss = torch.mean((sig - iv_t[idx]) ** 2)
            running += loss.item()
            if sobolev > 0:
                c = bs_call_torch(xb[:, 0], xb[:, 1], xb[:, 2], sig)
                g = torch.autograd.grad(c.sum(), xb, create_graph=True)[0][:, 0]
                loss = loss + sobolev * torch.mean((g - d_t[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

        model.eval()
        with torch.no_grad():
            err = model(Xv_t).numpy().astype(np.float64) - yv
        hist["epoch"].append(ep)
        hist["train_mse"].append(running / steps)              # här: MSE i implicit vol
        hist["val_rmse"].append(float(np.sqrt(np.mean(err**2))))
        hist["val_max"].append(float(np.max(np.abs(err))))
        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"epok {ep:4d}  iv-RMSE {np.sqrt(hist['train_mse'][-1]):.2e}  val-RMSE (pris) "
                  f"{hist['val_rmse'][-1]:.3e}  val-max {hist['val_max'][-1]:.3e}  ({time.time() - t0:.0f} s)",
                  flush=True)
    return hist, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--n_train", type=int, default=2**18)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--sobolev", type=float, default=1.0)
    ap.add_argument("--hard", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iv", action="store_true", help="lär implicit volatilitet i stället för pris")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(RES, "heston_iv.pt" if a.iv else "heston_modell.pt")

    X, y, dy, Xv, yv = load_or_make(a.n_train, a.seed, a.hard)
    torch.manual_seed(a.seed)
    extra = {}
    if a.iv:
        iv, ok = iv_labels(X, y)
        print(f"implicit vol: {ok.sum()} av {len(ok)} punkter används ({(~ok).sum()} har för litet vega)")
        model = IVNet(a.width, a.depth, WIDE_DOMAIN)
        hist, secs = fit_iv(model, X[ok], iv[ok], dy[ok], Xv, yv, a.epochs, a.batch, a.lr, a.sobolev)
        extra = {"iv": True, "n_used": int(ok.sum())}
    else:
        model = PricerMLP(a.width, a.depth, WIDE_DOMAIN)
        hist, secs = fit(model, X, y, dy, Xv, yv, a.epochs, a.batch, a.lr, a.sobolev)
    cfg = dict(width=a.width, depth=a.depth, n_train=a.n_train, epochs=a.epochs, batch=a.batch, lr=a.lr,
               sobolev=a.sobolev, hard=a.hard, seed=a.seed, wide=True, train_seconds=secs,
               n_params=sum(p.numel() for p in model.parameters()), **extra)
    save(model, hist, cfg, out)
    print(f"Sparad: {os.path.relpath(out, HERE)}  ({cfg['n_params']} parametrar, {secs:.0f} s)")
