"""
Black-Scholes för europeisk köpoption: exakt formel, greker, Monte Carlo och Crank-Nicolson.

Priset är homogent av grad 1 i (S, K), så allt räknas med m = S/K och c = C/K.
tau = T - t är tiden kvar till förfall.
"""
import numpy as np
from scipy.linalg import solve_banded
from scipy.interpolate import CubicSpline
from scipy.stats import norm

# Domänen som nätet tränas och testas på
DOMAIN = {
    "m": (0.60, 1.40),
    "tau": (0.10, 2.00),
    "r": (0.00, 0.08),
    "sigma": (0.10, 0.50),
}
FEATURES = ["m", "tau", "r", "sigma"]


def sample_domain(n, rng):
    lo = np.array([DOMAIN[f][0] for f in FEATURES])
    hi = np.array([DOMAIN[f][1] for f in FEATURES])
    return lo + (hi - lo) * rng.random((n, len(FEATURES)))


def _d1_d2(m, tau, r, sigma):
    s = sigma * np.sqrt(tau)
    d1 = (np.log(m) + (r + 0.5 * sigma**2) * tau) / s
    return d1, d1 - s


def call_price(m, tau, r, sigma):
    d1, d2 = _d1_d2(m, tau, r, sigma)
    return m * norm.cdf(d1) - np.exp(-r * tau) * norm.cdf(d2)


def call_greeks(m, tau, r, sigma):
    # Derivator av c(m, tau, r, sigma). Med C = K c gäller delta = dc/dm, gamma = c_mm / K,
    # vega = K dc/dsigma, theta = -K dc/dtau och rho = K dc/dr.
    d1, d2 = _d1_d2(m, tau, r, sigma)
    pdf1 = norm.pdf(d1)
    disc = np.exp(-r * tau)
    sq = np.sqrt(tau)
    return {
        "delta": norm.cdf(d1),
        "gamma": pdf1 / (m * sigma * sq),
        "vega": m * pdf1 * sq,
        "dtau": m * pdf1 * sigma / (2 * sq) + r * disc * norm.cdf(d2),
        "rho": tau * disc * norm.cdf(d2),
    }


def implied_vol(c, m, tau, r, lo=1e-4, hi=5.0, iters=80):
    """Implicit volatilitet med bisektion (vektoriserad). NaN om priset bryter arbitragegränserna."""
    c, m, tau, r = np.broadcast_arrays(*(np.asarray(z, float) for z in (c, m, tau, r)))
    ok = (c > np.maximum(m - np.exp(-r * tau), 0)) & (c < m)
    a, b = np.full(c.shape, lo), np.full(c.shape, hi)
    for _ in range(iters):
        mid = 0.5 * (a + b)
        below = call_price(m, tau, r, mid) < c
        a = np.where(below, mid, a)
        b = np.where(below, b, mid)
    return np.where(ok, 0.5 * (a + b), np.nan)


def bs_pde_residual(c, dc_dm, d2c_dm2, dc_dtau, m, r, sigma):
    # BS-PDE:n i (m, tau): -c_tau + 1/2 sigma^2 m^2 c_mm + r m c_m - r c = 0
    return -dc_dtau + 0.5 * sigma**2 * m**2 * d2c_dm2 + r * m * dc_dm - r * c


def monte_carlo_call(m, tau, r, sigma, n_paths, rng):
    """
    Riskneutral Monte Carlo med antitetiska variabler och importance sampling.
    Normalfördelningen flyttas med mu så att medianbanan hamnar vid strike; annars
    träffar nästan inga banor för optioner långt ut ur pengarna. Returnerar (skattning, std.fel).
    """
    w = rng.standard_normal(n_paths // 2)
    drift = (r - 0.5 * sigma**2) * tau
    vol = sigma * np.sqrt(tau)
    mu = max(-np.log(m) / vol, 0.0)

    def weighted_payoff(w_):
        z = mu + w_
        return np.maximum(m * np.exp(drift + vol * z) - 1.0, 0.0) * np.exp(-mu * w_ - 0.5 * mu**2)

    pairs = np.exp(-r * tau) * 0.5 * (weighted_payoff(w) + weighted_payoff(-w))
    return pairs.mean(), pairs.std(ddof=1) / np.sqrt(len(pairs))


def crank_nicolson_call(m_eval, tau, r, sigma, n_x=800, n_t=400):
    """
    Löser PDE:n numeriskt i x = ln(m):  c_tau = 1/2 s^2 c_xx + (r - s^2/2) c_x - r c.
    Crank-Nicolson med fyra implicita Euler-steg i början (Rannacher) eftersom
    payoffen har en kink. Rand: c(-L) = 0 och c(L) = e^L - e^{-r tau}.
    """
    L = max(6.0 * sigma * np.sqrt(tau), 1.5)
    x = np.linspace(-L, L, n_x + 1)
    h = x[1] - x[0]
    a = 0.5 * sigma**2 / h**2 - (r - 0.5 * sigma**2) / (2 * h)   # c_{j-1}
    b = -sigma**2 / h**2 - r                                      # c_j
    d = 0.5 * sigma**2 / h**2 + (r - 0.5 * sigma**2) / (2 * h)   # c_{j+1}
    n_in = n_x - 1

    def step(c, dt, theta, t_new):
        # (I - theta dt A) c_ny = (I + (1 - theta) dt A) c + randterm
        ci = c[1:-1]
        rhs = ci + (1 - theta) * dt * (a * c[:-2] + b * ci + d * c[2:])
        right_new = np.exp(L) - np.exp(-r * t_new)
        rhs[-1] += theta * dt * d * right_new
        ab = np.zeros((3, n_in))
        ab[0, 1:] = -theta * dt * d
        ab[1, :] = 1 - theta * dt * b
        ab[2, :-1] = -theta * dt * a
        out = np.empty_like(c)
        out[0], out[-1] = 0.0, right_new
        out[1:-1] = solve_banded((1, 1), ab, rhs)
        return out

    c = np.maximum(np.exp(x) - 1.0, 0.0)
    dt = tau / n_t
    t = 0.0
    for _ in range(4):
        t += dt / 4
        c = step(c, dt / 4, 1.0, t)
    for _ in range(n_t - 1):
        t += dt
        c = step(c, dt, 0.5, t)
    return CubicSpline(x, c)(np.log(m_eval))
