"""
Lokal dashboard för nätverken. Starta med

    streamlit run app.py

och öppna adressen som skrivs ut (normalt http://localhost:8501).
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from black_scholes import DOMAIN as BS_DOMAIN, bs_pde_residual, call_greeks, call_price, implied_vol
from calibrate import MATURITIES, PARAMS, R as CAL_R, STRIKES, NNPricer, calibrate, cos_pricer, iv_rmse, make_market
from heston import DOMAIN as H_DOMAIN, FEATURES as H_FEATURES, cos_call
from nn_model import load_model, nn_price, nn_price_and_derivatives

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultat")
# Mellanstegen ligger i resultat/arkiv/ och finns beskrivna i README.
BS_MODELS = {
    "Första nätet (bara priser)": ("modell.pt", "standard"),
    "Bästa nätet": ("modell_bast.pt", "bast"),
}
BS_MODELS = {k: v for k, v in BS_MODELS.items() if os.path.exists(os.path.join(RES, v[0]))}
H_MODELS = {"Pris-nät": ("heston_modell_400.pt", "_400"), "IV-nät": ("heston_iv.pt", "_iv")}
H_MODELS = {k: v for k, v in H_MODELS.items() if os.path.exists(os.path.join(RES, v[0]))}
ORE = 1e4   # c = C/K  ->  öre per option när K = 100 kr

st.set_page_config(page_title="NN-prissättare", layout="wide")
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})


@st.cache_resource
def get_model(fname):
    return load_model(os.path.join(RES, fname))[0].double()


def read_json(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def slider(label, key, lo, hi, value, step, fmt="%.2f"):
    return st.slider(label, float(lo), float(hi), float(value), float(step), format=fmt, key=key)


@st.cache_resource
def get_nn_pricer():
    return NNPricer()


@st.cache_data
def run_calibration(p_true, noise, seed, with_cos):
    mkt = make_market(np.array(p_true), noise, np.random.default_rng(seed))
    out = {"mkt": mkt}
    out["nn"], out["t_nn"], _ = calibrate(get_nn_pricer(), mkt)
    out["fit_nn"] = iv_rmse(out["nn"], mkt)
    if with_cos:
        out["cos"], out["t_cos"], _ = calibrate(cos_pricer, mkt)
        out["fit_cos"] = iv_rmse(out["cos"], mkt)
    return out


st.title("Neuralt nät som optionsprissättare")
tab_bs, tab_h, tab_cal, tab_ver = st.tabs(["Black-Scholes", "Heston", "Kalibrering", "Verifiering"])

# ---------------------------------------------------------------------------
with tab_bs:
    left, right = st.columns([1, 3], gap="large")
    with left:
        name = st.selectbox("Modell", list(BS_MODELS), index=len(BS_MODELS) - 1)
        ext = st.checkbox("Tillåt värden utanför träningsdomänen")
        rng_ = {"m": (0.3, 2.0), "tau": (0.02, 4.0), "r": (-0.02, 0.15), "sigma": (0.05, 0.9)} if ext else BS_DOMAIN
        m = slider("S/K", "bs_m", *rng_["m"], 1.0, 0.005, "%.3f")
        tau = slider("Löptid (år)", "bs_tau", *rng_["tau"], 0.5, 0.01)
        r = slider("Ränta", "bs_r", *rng_["r"], 0.03, 0.0025, "%.4f")
        sigma = slider("Volatilitet", "bs_sigma", *rng_["sigma"], 0.25, 0.005, "%.3f")
        greek = st.radio("Visa", ["Delta", "Gamma", "Vega", "Theta", "PDE-residual"], horizontal=True)
        inside = all(BS_DOMAIN[k][0] <= v <= BS_DOMAIN[k][1] for k, v in
                     dict(m=m, tau=tau, r=r, sigma=sigma).items())
        if not inside:
            st.warning("Punkten ligger utanför träningsdomänen, nätet extrapolerar.")

    model = get_model(BS_MODELS[name][0])
    x0 = np.array([[m, tau, r, sigma]])
    c_nn = nn_price(model, x0)[0]
    c_ex = call_price(m, tau, r, sigma)
    iv = float(implied_vol(c_nn, m, tau, r))

    with right:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Nätets pris (K = 100)", f"{100 * c_nn:.4f} kr")
        k2.metric("Exakt pris", f"{100 * c_ex:.4f} kr")
        k3.metric("Fel", f"{(c_nn - c_ex) * ORE:+.2f} öre")
        k4.metric("Fel i implicit vol", "saknas" if np.isnan(iv) else f"{100 * (iv - sigma):+.3f} volp.")

        ms = np.linspace(*rng_["m"], 201)
        X = np.column_stack([ms, np.full_like(ms, tau), np.full_like(ms, r), np.full_like(ms, sigma)])
        nn = nn_price_and_derivatives(model, X)
        ex = call_greeks(*X.T)
        ex["c"] = call_price(*X.T)

        fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
        ax[0, 0].plot(ms, 100 * ex["c"], "k", lw=1, label="exakt")
        ax[0, 0].plot(ms, 100 * nn["c"], "--", color="C0", label="nät")
        ax[0, 0].plot(ms, 100 * np.maximum(ms - np.exp(-r * tau), 0), color="0.6", lw=0.8, label="arbitragegräns")
        ax[0, 0].set_title("Pris (kr, K = 100)")
        ax[0, 0].legend()
        ax[0, 1].plot(ms, (nn["c"] - ex["c"]) * ORE, color="C0")
        ax[0, 1].axhline(0, color="0.5", lw=0.8)
        ax[0, 1].set_title("Fel, nät - exakt (öre)")

        if greek == "PDE-residual":
            R = bs_pde_residual(nn["c"], nn["delta"], nn["gamma"], nn["dtau"], ms, r, sigma)
            ax[1, 0].plot(ms, R, color="C0", label="nät")
            ax[1, 0].axhline(0, color="k", lw=1, label="exakt (R = 0)")
        else:
            key = {"Delta": "delta", "Gamma": "gamma", "Vega": "vega", "Theta": "dtau"}[greek]
            f = (lambda v: -100 * v / 365) if greek == "Theta" else (lambda v: v)
            ax[1, 0].plot(ms, f(ex[key]), "k", lw=1, label="exakt")
            ax[1, 0].plot(ms, f(nn[key]), "--", color="C0", label="nät")
        ax[1, 0].set_title(greek + (" (kr per dag)" if greek == "Theta" else ""))
        ax[1, 0].legend()

        mg, tg = np.linspace(*rng_["m"], 60), np.linspace(*rng_["tau"], 40)
        M, T = np.meshgrid(mg, tg)
        Xh = np.column_stack([M.ravel(), T.ravel(), np.full(M.size, r), np.full(M.size, sigma)])
        E = np.abs(nn_price(model, Xh) - call_price(*Xh.T)).reshape(M.shape) * ORE
        im = ax[1, 1].pcolormesh(M, T, np.log10(np.maximum(E, 1e-3)), cmap="viridis", shading="auto")
        cb = fig.colorbar(im, ax=ax[1, 1])
        cb.set_label("log10 |fel| (öre)")
        if ext:
            ax[1, 1].add_patch(plt.Rectangle((BS_DOMAIN["m"][0], BS_DOMAIN["tau"][0]),
                                             np.ptp(BS_DOMAIN["m"]), np.ptp(BS_DOMAIN["tau"]), fill=False, ec="w"))
        ax[1, 1].plot(m, tau, "wo", mec="k")
        ax[1, 1].set_title(f"|fel| över S/K och löptid (r = {r:.3f}, σ = {sigma:.3f})")
        ax[1, 1].set_ylabel("löptid")
        for a in ax.flat:
            a.set_xlabel("S/K")
            if a is not ax[1, 1]:
                a.axvline(m, color="0.5", lw=0.6)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ---------------------------------------------------------------------------
with tab_h:
    if not H_MODELS:
        st.info("Heston-nätet finns inte än. Kör train_heston.py först.")
    else:
        left, right = st.columns([1, 3], gap="large")
        D = H_DOMAIN
        with left:
            hname = st.selectbox("Modell", list(H_MODELS), index=len(H_MODELS) - 1, key="h_model")
            tau = slider("Löptid (år)", "h_tau", *D["tau"], 0.5, 0.01)
            r = slider("Ränta", "h_r", *D["r"], 0.02, 0.0025, "%.4f")
            v0 = slider("v0 (startvarians)", "h_v0", *D["v0"], 0.04, 0.005, "%.3f")
            kappa = slider("kappa", "h_kappa", *D["kappa"], 2.0, 0.1)
            theta = slider("theta (långsiktig varians)", "h_theta", *D["theta"], 0.04, 0.005, "%.3f")
            xi = slider("xi (vol of vol)", "h_xi", *D["xi"], 0.5, 0.01)
            rho = slider("rho", "h_rho", *D["rho"], -0.7, 0.01)
            st.caption(f"Feller-villkoret 2·kappa·theta ≥ xi²: {2 * kappa * theta:.3f} vs {xi**2:.3f} "
                       f"({'uppfyllt' if 2 * kappa * theta >= xi**2 else 'inte uppfyllt'})")

        hmodel = get_model(H_MODELS[hname][0])
        ks = np.linspace(1 / D["m"][1] + 0.005, 1 / D["m"][0] - 0.005, 81)     # K/S inom domänen
        ms = 1 / ks
        params = [tau, r, v0, kappa, theta, xi, rho]
        X = np.column_stack([ms] + [np.full_like(ms, p) for p in params])
        c_cos = cos_call(*X.T)
        c_nn = nn_price(hmodel, X)
        # implicit vol bara där tidsvärdet är minst 0,5 kr, annars är vega nästan noll
        good = c_cos - np.maximum(ms - np.exp(-r * tau), 0) >= 0.005
        iv_cos = np.where(good, implied_vol(c_cos, ms, tau, r), np.nan)
        iv_nn = np.where(good, implied_vol(c_nn, ms, tau, r), np.nan)
        i0 = np.argmin(np.abs(ks - 1.0))

        with right:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Nätets pris ATM (K = 100)", f"{100 * c_nn[i0]:.4f} kr")
            k2.metric("COS-pris ATM", f"{100 * c_cos[i0]:.4f} kr")
            k3.metric("Största fel längs strike", f"{np.max(np.abs(c_nn - c_cos)) * ORE:.2f} öre")
            d_iv = np.abs(iv_nn - iv_cos)
            k4.metric("Största fel i implicit vol", "saknas" if np.all(np.isnan(d_iv)) else
                      f"{100 * np.nanmax(d_iv):.3f} volp.", help="Strikes med tidsvärde minst 0,5 kr")

            fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
            ax[0].plot(ks, 100 * iv_cos, "k", lw=1, label="COS")
            ax[0].plot(ks, 100 * iv_nn, "--", color="C0", label="nät")
            ax[0].plot(ks, np.full_like(ks, 100 * np.sqrt(v0)), color="0.6", lw=0.8, label="sqrt(v0)")
            ax[0].set_title("Volatilitetsleende")
            ax[0].set_ylabel("implicit vol (%)")
            ax[0].legend()
            ax[1].plot(ks, (c_nn - c_cos) * ORE, color="C0")
            ax[1].axhline(0, color="0.5", lw=0.8)
            ax[1].set_title("Fel, nät - COS (öre)")
            for a in ax:
                a.set_xlabel("K/S")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            hs = read_json(f"heston_sammanfattning{H_MODELS[hname][1]}.json")
            if hs:
                t = hs["timing"]
                st.write(f"Tid per option: nätet {t['nn'] * 1e6:.1f} µs, COS {t['cos_ref'] * 1e6:.0f} µs, "
                         f"quad {t['quad'] * 1e3:.0f} ms, Monte Carlo {t['mc']:.1f} s.")

# ---------------------------------------------------------------------------
with tab_cal:
    if not os.path.exists(os.path.join(RES, "heston_modell_400.pt")):
        st.info("Kalibreringen använder heston_modell_400.pt. Kör train_heston.py --epochs 400 först.")
    else:
        st.caption("En volatilitetsyta med 9 strikes och 5 löptider räknas fram med COS från parametrarna "
                   "till vänster, plus brus. Sedan kalibreras Heston mot ytan, en gång med nätet som "
                   "prissättare och en gång med COS.")
        left, right = st.columns([1, 3], gap="large")
        D = H_DOMAIN
        with left:
            p_true = (
                slider("v0", "c_v0", *D["v0"], 0.04, 0.005, "%.3f"),
                slider("kappa", "c_kappa", *D["kappa"], 1.5, 0.1),
                slider("theta", "c_theta", *D["theta"], 0.06, 0.005, "%.3f"),
                slider("xi", "c_xi", *D["xi"], 0.6, 0.01),
                slider("rho", "c_rho", *D["rho"], -0.7, 0.01),
            )
            noise = slider("Brus i ytan (volpunkter)", "c_noise", 0.0, 1.0, 0.25, 0.05)
            seed = st.number_input("Slumpfrö för bruset", 0, 9999, 1, key="c_seed")
            with_cos = st.checkbox("Kalibrera även med COS", value=True, key="c_cos")
        res = run_calibration(tuple(float(v) for v in p_true), float(noise), int(seed), with_cos)

        with right:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Tid med nätet", f"{res['t_nn']:.3f} s")
            k2.metric("Passning, nätets parametrar", f"{res['fit_nn']:.3f} volp.",
                      help="Kalibrerade parametrar omprissatta med COS, RMSE mot ytan")
            if with_cos:
                k3.metric("Tid med COS", f"{res['t_cos']:.2f} s")
                k4.metric("Passning, COS-parametrar", f"{res['fit_cos']:.3f} volp.")
            table = {"sant": p_true, "nät": res["nn"]}
            if with_cos:
                table["COS"] = res["cos"]
            st.dataframe(pd.DataFrame(table, index=PARAMS).style.format(precision=4), width="content")

            mkt = res["mkt"]
            ks = np.linspace(STRIKES[0], STRIKES[-1], 41)
            fits = [(res["nn"], "-", "C0", "nät")] + ([(res["cos"], "--", "C1", "COS")] if with_cos else [])
            fig, ax = plt.subplots(1, len(MATURITIES), figsize=(13, 3.2), sharey=True)
            for a, T in zip(ax, MATURITIES):
                sel = np.isclose(mkt["tau"], T)
                a.plot(1 / mkt["m"][sel], 100 * mkt["iv"][sel], "ko", ms=4, label="yta")
                for p, ls, col, label in fits:
                    iv_fit = implied_vol(cos_call(1 / ks, T, CAL_R, *p), 1 / ks, T, CAL_R)
                    a.plot(ks, 100 * iv_fit, ls, color=col, label=f"kalibrerad med {label}")
                a.set_title(f"τ = {T}")
                a.set_xlabel("K/S")
            ax[0].set_ylabel("implicit vol (%)")
            ax[0].legend(fontsize=7)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Kurvorna är de kalibrerade parametrarna prissatta med COS, alltså vad parametrarna "
                       "faktiskt ger för priser.")

        cal = read_json("kalibrering.json")
        if cal:
            st.subheader("40 slumpade ytor")
            rows = {}
            for nz in sorted({c["noise"] for c in cal["cases"]}):
                cs = [c for c in cal["cases"] if c["noise"] == nz]
                T = np.array([c["true"] for c in cs])
                for who, name in (("nn", "nät"), ("cos", "COS")):
                    E = np.abs(np.array([c[who] for c in cs]) - T)
                    row = {p: float(np.median(E[:, j])) for j, p in enumerate(PARAMS)}
                    row["passning (volp.)"] = float(np.median([c[f"fit_{who}"] for c in cs]))
                    row["tid (s)"] = float(np.median([c[f"t_{who}"] for c in cs]))
                    rows[f"{name}, brus {nz:g} volp."] = row
            st.dataframe(pd.DataFrame(rows).T.style.format(precision=4), width="stretch")
            st.caption("Median av absolutfelet i varje parameter, passning och tid per kalibrering.")
            img = os.path.join(RES, "fig_kalibrering.png")
            if os.path.exists(img):
                st.image(img)

# ---------------------------------------------------------------------------
with tab_ver:
    st.subheader("Black-Scholes-näten mot exakt lösning")
    rows = {}
    for label, (_, tag) in BS_MODELS.items():
        s = read_json(f"sammanfattning_{tag}.json")
        if not s:
            continue
        e = s["errors"]
        rows[label] = {
            "MAE (öre)": e["MAE"] * ORE, "RMSE (öre)": e["RMSE"] * ORE, "max (öre)": e["max"] * ORE,
            "delta, rel. fel (%)": 100 * s["greeks"]["delta"]["rel_MAE"],
            "gamma, rel. fel (%)": 100 * s["greeks"]["gamma"]["rel_MAE"],
            "PDE-residual (%)": 100 * s["pde"]["rel"],
            "arbitragebrott, medel över villkoren (% av punkter)":
                100 * np.mean([a["frac"] for a in s["arbitrage"].values()]),
            "gamma < 0 (% av punkter)": 100 * s["arbitrage"]["gamma >= 0"]["frac"],
            "maximumprincipen": "håller" if s["max_principle"]["all_ok"] else "bryts",
            "träningstid (min)": s["config"]["train_seconds"] / 60,
        }
    if rows:
        st.dataframe(pd.DataFrame(rows).T.style.format(precision=2), width="stretch")
    st.caption("10^6 testpunkter i träningsdomänen. Fel i öre per option med K = 100 kr.")

    st.subheader("Heston-näten mot COS")
    rows = {}
    for label, (_, sfx) in H_MODELS.items():
        s = read_json(f"heston_sammanfattning{sfx}.json")
        if not s:
            continue
        e = s["errors"]
        rows[label] = {
            "MAE (öre)": e["MAE"] * ORE, "RMSE (öre)": e["RMSE"] * ORE, "max (öre)": e["max"] * ORE,
            "impl. vol, medel (volp.)": s["iv"]["MAE"], "impl. vol, 99 % (volp.)": s["iv"]["p99"],
            "PDE-residual (%)": 100 * s["pde"]["rel"],
            "pris under nedre gränsen (% av punkter)": 100 * s["arbitrage"]["c >= max(m - e^-r tau, 0)"]["frac"],
            "arbitragebrott, medel över villkoren (%)": 100 * np.mean([a["frac"] for a in s["arbitrage"].values()]),
            "träningstid (min)": s["config"]["train_seconds"] / 60,
        }
    if rows:
        st.dataframe(pd.DataFrame(rows).T.style.format(precision=2), width="stretch")
    st.caption("10^5 testpunkter. Implicit vol för optioner med tidsvärde minst 0,5 kr.")

    ex = read_json("sammanfattning_exakt.json")
    hr = read_json("heston_referens.json")
    st.subheader("Referenslösningarna")
    if ex:
        n_ok = sum(x["ok"] for x in ex["symbolic"])
        st.markdown(f"- Black-Scholes, sympy: {n_ok} av {len(ex['symbolic'])} identiteter blir exakt 0\n"
                    f"- Black-Scholes, Monte Carlo: p = {ex['mc']['p_value']:.2f}")
    if hr:
        st.markdown(f"- Heston, COS mot publicerat värde: {hr['benchmark']['cos']:.9f} (ska vara 5.785155450)\n"
                    f"- Heston, COS mot quad på {hr['cos_vs_quad']['n']} punkter: största skillnad "
                    f"{hr['cos_vs_quad']['max']:.1e}\n"
                    f"- Heston, PDE-residual för COS: {hr['pde_cos']['mean']:.1e}")
    sfx = list(H_MODELS.values())[-1][1] if H_MODELS else ""
    for img in ("fig_konvergens.png", f"fig_heston_leende{sfx}.png"):
        if os.path.exists(os.path.join(RES, img)):
            st.image(os.path.join(RES, img))
