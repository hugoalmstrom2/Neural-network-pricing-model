"""
Kalibrering av Heston mot en volatilitetsyta, med nätet som prissättare och med COS som jämförelse.

  python calibrate.py              # 40 syntetiska ytor, utan brus och med 0,25 volpunkters brus

Ytan skapas med COS från kända parametrar, så facit är känt. Parametrarna (v0, kappa, theta, xi, rho)
anpassas så att de vega-viktade prisfelen blir minimala, vilket ungefär är felen i implicit vol.
"""
import json
import os
import time

import numpy as np
import torch
from scipy.optimize import least_squares

from black_scholes import call_greeks, call_price, implied_vol
from heston import DOMAIN, cos_call
from nn_model import load_model

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "resultat")
STRIKES = np.array([0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2])     # K/S
MATURITIES = np.array([0.25, 0.5, 1.0, 1.5, 2.0])
R = 0.02
PARAMS = ["v0", "kappa", "theta", "xi", "rho"]
LO = np.array([DOMAIN[p][0] for p in PARAMS])
HI = np.array([DOMAIN[p][1] for p in PARAMS])
X0 = np.array([0.05, 2.0, 0.05, 0.5, -0.5])                               # startgissning


def grid():
    K, T = np.meshgrid(STRIKES, MATURITIES)
    return 1 / K.ravel(), T.ravel()                                       # m = S/K, tau


class NNPricer:
    def __init__(self, path=os.path.join(RES, "heston_iv.pt")):
        self.model = load_model(path)[0].double()

    def _inputs(self, m, tau, p):
        return torch.from_numpy(np.column_stack([m, tau, np.full_like(m, R)] + [np.full_like(m, v) for v in p]))

    def __call__(self, m, tau, p):
        with torch.inference_mode():
            return self.model(self._inputs(m, tau, p)).numpy()

    def jac(self, m, tau, p):
        # Alla optioner har samma parametrar, så dc_i/dp_j är gradienten av c_i med avseende
        # på indatakolumn 3 + j i rad i. Ett enda bakåtpass ger hela Jacobianen.
        X = self._inputs(m, tau, p).requires_grad_(True)
        g = torch.autograd.grad(self.model(X).sum(), X)[0]
        return g[:, 3:].numpy()


def cos_pricer(m, tau, p):
    return cos_call(m, tau, R, *p)


def make_market(p_true, noise_vp, rng):
    """Marknadsyta: COS-priser omräknade till implicit vol, plus normalfördelat brus i volpunkter."""
    m, tau = grid()
    iv = implied_vol(cos_call(m, tau, R, *p_true), m, tau, R)
    iv_mkt = iv + noise_vp / 100 * rng.standard_normal(iv.shape)
    return {"m": m, "tau": tau, "iv": iv_mkt, "c": call_price(m, tau, R, iv_mkt),
            "vega": call_greeks(m, tau, R, iv_mkt)["vega"]}


def calibrate(pricer, mkt, x0=X0, use_grad=True):
    # residual = prisfel / vega, ungefär felet i implicit vol
    m, tau, vega = mkt["m"], mkt["tau"], mkt["vega"]
    f = lambda p: (pricer(m, tau, p) - mkt["c"]) / vega
    if use_grad and hasattr(pricer, "jac"):
        jac = lambda p: pricer.jac(m, tau, p) / vega[:, None]     # exakt Jacobian från nätet
    else:
        jac = "2-point"                                             # finita differenser
    t0 = time.time()
    res = least_squares(f, x0, jac=jac, bounds=(LO, HI), x_scale=HI - LO, diff_step=1e-5)
    return res.x, time.time() - t0, res.nfev


def iv_rmse(p, mkt):
    """Hur väl parametrarna passar ytan när de prissätts med COS, i volpunkter."""
    iv = implied_vol(cos_call(mkt["m"], mkt["tau"], R, *p), mkt["m"], mkt["tau"], R)
    return 100 * np.sqrt(np.nanmean((iv - mkt["iv"]) ** 2))


def random_params(n, rng):
    # inre 80 % av domänen så att sanna parametrar inte ligger på gränsen
    lo, hi = LO + 0.1 * (HI - LO), HI - 0.1 * (HI - LO)
    return lo + (hi - lo) * rng.random((n, len(PARAMS)))


if __name__ == "__main__":
    nn = NNPricer()
    rng = np.random.default_rng(42)
    truths = random_params(40, rng)
    out, lines = {"params": PARAMS, "cases": []}, []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    for noise in (0.0, 0.25):
        rows = []
        for i, p_true in enumerate(truths):
            mkt = make_market(p_true, noise, np.random.default_rng(1000 + i))
            p_nn, t_nn, n_nn = calibrate(nn, mkt)
            p_cos, t_cos, n_cos = calibrate(cos_pricer, mkt)
            rows.append({"noise": noise, "true": p_true.tolist(), "nn": p_nn.tolist(), "cos": p_cos.tolist(),
                         "t_nn": t_nn, "t_cos": t_cos, "nfev_nn": n_nn, "nfev_cos": n_cos,
                         "fit_nn": iv_rmse(p_nn, mkt), "fit_cos": iv_rmse(p_cos, mkt),
                         "fit_true": iv_rmse(p_true, mkt)})
        out["cases"] += rows

        T = np.array([r["true"] for r in rows])
        E_nn = np.abs(np.array([r["nn"] for r in rows]) - T)
        E_cos = np.abs(np.array([r["cos"] for r in rows]) - T)
        log(f"Brus {noise} volpunkter, {len(rows)} ytor")
        log(f"   {'':6} {'median |fel|, nät':>18} {'median |fel|, COS':>18} {'90 %, nät':>10} {'90 %, COS':>10}")
        for j, name in enumerate(PARAMS):
            log(f"   {name:6} {np.median(E_nn[:, j]):18.4f} {np.median(E_cos[:, j]):18.4f} "
                f"{np.quantile(E_nn[:, j], 0.9):10.4f} {np.quantile(E_cos[:, j], 0.9):10.4f}")
        f_nn = [r["fit_nn"] for r in rows]
        f_cos = [r["fit_cos"] for r in rows]
        f_true = [r["fit_true"] for r in rows]
        log(f"   passning (COS-omprissatt), median volpunkter: nät {np.median(f_nn):.3f}, "
            f"COS {np.median(f_cos):.3f}, sanna parametrar {np.median(f_true):.3f}")
        t_nn = np.median([r["t_nn"] for r in rows])
        t_cos = np.median([r["t_cos"] for r in rows])
        log(f"   tid per kalibrering, median: nät {t_nn:.3f} s, COS {t_cos:.2f} s ({t_cos / t_nn:.0f} gånger)")
        log()

    with open(os.path.join(RES, "kalibrering.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(RES, "kalibrering.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Figur: sanna mot kalibrerade parametrar (med brus) och ett exempel på passning
    from plot_style import apply_style, plt
    apply_style()
    rows = [r for r in out["cases"] if r["noise"] > 0]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
    for j, (ax, name) in enumerate(zip(axes, PARAMS)):
        t = [r["true"][j] for r in rows]
        ax.plot(t, [r["cos"][j] for r in rows], "x", color="C1", label="COS")
        ax.plot(t, [r["nn"][j] for r in rows], "o", mfc="none", color="C0", label="nät")
        ax.plot(DOMAIN[name], DOMAIN[name], color="0.6", lw=0.8)
        ax.set_title(name)
        ax.set_xlabel("sant värde")
    axes[0].set_ylabel("kalibrerat värde")
    axes[0].legend()
    fig.suptitle("Kalibrerade mot sanna parametrar, 40 ytor med 0,25 volpunkters brus")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_kalibrering.png"))
    plt.close(fig)
    print("Sparat i resultat/")
