"""
Träning av Black-Scholes-nätet.

  python train.py                                   # grundmodell, bara priser
  python train.py --sobolev 1 --out resultat/modell_sobolev.pt
  python train.py --sobolev 1 --wide --hard 0.3 --epochs 400 --out resultat/modell_forbattrad.pt

--wide  tränar på en något större domän än testdomänen (kanterna blir bättre)
--hard  andel extra träningspunkter med kort löptid, låg vol och nära ATM
--shape vikt för straff på brott mot arbitragevillkoren
--softplus  utgången kan inte bli negativ (softplus(beta y)/beta, beta = 100)

Bästa modellen:
  python train.py --sobolev 1 --wide --hard 0.3 --shape 1 --softplus --epochs 400 --out resultat/modell_bast.pt
"""
import argparse
import os
import time

import numpy as np
import torch

from black_scholes import DOMAIN, call_greeks, call_price
from nn_model import IVNet, PricerMLP

HERE = os.path.dirname(os.path.abspath(__file__))
WIDE_DOMAIN = {"m": (0.55, 1.45), "tau": (0.08, 2.1), "r": (-0.005, 0.085), "sigma": (0.08, 0.52)}


def sample_points(n, rng, domain, hard=0.0):
    lo = np.array([v[0] for v in domain.values()])
    hi = np.array([v[1] for v in domain.values()])
    X = lo + (hi - lo) * rng.random((n, len(domain)))
    k = int(hard * n)
    if k:
        # där felkartan visade störst fel
        X[:k, 0] = rng.uniform(0.85, 1.15, k)
        X[:k, 1] = rng.uniform(domain["tau"][0], 0.5, k)
        X[:k, 3] = rng.uniform(domain["sigma"][0], 0.3, k)
    return X


def make_dataset(n, seed, wide=False, hard=0.0):
    rng = np.random.default_rng(seed)
    X = sample_points(n, rng, WIDE_DOMAIN if wide else DOMAIN, hard)
    return X, call_price(*X.T), call_greeks(*X.T)["delta"]


def bs_shape_penalty(model, n=1024):
    """
    Kvadrerade brott mot arbitragevillkoren i n slumpade punkter i testdomänen:
    max(m - e^(-r tau), 0) <= c <= m, 0 <= delta <= 1, gamma >= 0, c_tau >= 0, rho >= 0, vega >= 0.
    """
    lo = torch.tensor([v[0] for v in DOMAIN.values()])
    hi = torch.tensor([v[1] for v in DOMAIN.values()])
    x = (lo + (hi - lo) * torch.rand(n, 4)).requires_grad_(True)
    c = model(x)
    g = torch.autograd.grad(c.sum(), x, create_graph=True)[0]
    gamma = torch.autograd.grad(g[:, 0].sum(), x, create_graph=True)[0][:, 0]
    m, tau, r = x[:, 0], x[:, 1], x[:, 2]
    relu = torch.relu
    viol = [relu(torch.clamp(m - torch.exp(-r * tau), min=0) - c), relu(c - m),
            relu(-g[:, 0]), relu(g[:, 0] - 1), relu(-gamma), relu(-g[:, 1]), relu(-g[:, 2]), relu(-g[:, 3])]
    return sum(torch.mean(v**2) for v in viol)


def fit(model, X, y, dy, Xv, yv, epochs=200, batch=4096, lr=2e-3, sobolev=0.0, verbose=True,
        penalty=None, shape=0.0):
    """Adam + cosine-schema. dy = dc/dx_0 (delta) används om sobolev > 0, penalty(model) om shape > 0."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    d_t = torch.tensor(dy, dtype=torch.float32) if dy is not None else None
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
            xb = X_t[idx]
            if sobolev > 0:
                xb.requires_grad_(True)
            pred = model(xb)
            loss = torch.mean((pred - y_t[idx]) ** 2)
            running += loss.item()
            if sobolev > 0:
                g = torch.autograd.grad(pred.sum(), xb, create_graph=True)[0][:, 0]
                loss = loss + sobolev * torch.mean((g - d_t[idx]) ** 2)
            if shape > 0:
                loss = loss + shape * penalty(model)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

        model.eval()
        with torch.no_grad():
            err = model(Xv_t).numpy().astype(np.float64) - yv
        hist["epoch"].append(ep)
        hist["train_mse"].append(running / steps)
        hist["val_rmse"].append(float(np.sqrt(np.mean(err**2))))
        hist["val_max"].append(float(np.max(np.abs(err))))
        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"epok {ep:4d}  train-MSE {hist['train_mse'][-1]:.3e}  val-RMSE {hist['val_rmse'][-1]:.3e}  "
                  f"val-max {hist['val_max'][-1]:.3e}  ({time.time() - t0:.0f} s)", flush=True)
    return hist, time.time() - t0


def train(width=128, depth=4, n_train=2**18, epochs=200, batch=4096, lr=2e-3, sobolev=0.0, seed=0,
          wide=False, hard=0.0, shape=0.0, out_act="none", beta=100.0, verbose=True):
    torch.manual_seed(seed)
    X, y, d = make_dataset(n_train, seed, wide, hard)
    Xv, yv, _ = make_dataset(2**14, seed + 1000)          # validering i testdomänen
    model = PricerMLP(width, depth, WIDE_DOMAIN if wide else DOMAIN, out_act, beta)
    hist, secs = fit(model, X, y, d, Xv, yv, epochs, batch, lr, sobolev, verbose, bs_shape_penalty, shape)
    config = dict(width=width, depth=depth, n_train=n_train, epochs=epochs, batch=batch, lr=lr,
                  sobolev=sobolev, seed=seed, wide=wide, hard=hard, shape=shape, out_act=out_act, beta=beta,
                  train_seconds=secs, n_params=sum(p.numel() for p in model.parameters()))
    return model, hist, config


def save(model, history, config, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "width": model.width, "depth": model.depth,
                "domain": model.domain, "out_act": getattr(model, "out_act", "none"),
                "beta": getattr(model, "beta", 1.0),
                "kind": "iv" if isinstance(model, IVNet) else "price",
                "history": history, "config": config}, path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--n_train", type=int, default=2**18)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--sobolev", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--hard", type=float, default=0.0)
    ap.add_argument("--shape", type=float, default=0.0)
    ap.add_argument("--softplus", action="store_true", help="utgång som inte kan bli negativ")
    ap.add_argument("--out", default=os.path.join(HERE, "resultat", "modell.pt"))
    a = ap.parse_args()
    model, hist, cfg = train(a.width, a.depth, a.n_train, a.epochs, a.batch, a.lr, a.sobolev, a.seed,
                             a.wide, a.hard, a.shape, "softplus" if a.softplus else "none")
    save(model, hist, cfg, a.out)
    print(f"Sparad: {os.path.relpath(a.out, HERE)}  ({cfg['n_params']} parametrar, {cfg['train_seconds']:.0f} s)")
