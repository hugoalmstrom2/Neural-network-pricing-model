"""
Heston-modellen: dS = rS dt + sqrt(v) S dW1,  dv = kappa(theta - v) dt + xi sqrt(v) dW2,  corr = rho.

Ingen sluten formel, men den karakteristiska funktionen är känd. Tre sätt att prissätta:
  cos_call   - COS-metoden (Fang & Oosterlee 2008), snabb och vektoriserad. Referens för nätet.
  quad_call  - Gil-Pelaez-integralen med scipy.quad. Långsam men oberoende av COS.
  mc_call    - Monte Carlo (Euler med full truncation). Används bara som rimlighetskontroll.

Samma normalisering som för Black-Scholes: m = S/K och c = C/K.
"""
import numpy as np
from scipy.integrate import quad

FEATURES = ["m", "tau", "r", "v0", "kappa", "theta", "xi", "rho"]
DOMAIN = {
    "m": (0.6, 1.4), "tau": (0.1, 2.0), "r": (0.0, 0.08),
    "v0": (0.01, 0.25),      # startvarians, vol 10-50 %
    "kappa": (0.5, 5.0),     # återgångshastighet
    "theta": (0.01, 0.25),   # långsiktig varians
    "xi": (0.1, 1.0),        # vol of vol
    "rho": (-0.9, 0.0),      # korrelation aktie/varians
}
WIDE_DOMAIN = {
    "m": (0.55, 1.45), "tau": (0.08, 2.1), "r": (-0.005, 0.085), "v0": (0.008, 0.27),
    "kappa": (0.4, 5.3), "theta": (0.008, 0.27), "xi": (0.08, 1.05), "rho": (-0.95, 0.05),
}


def sample(n, rng, domain=DOMAIN):
    lo = np.array([domain[f][0] for f in FEATURES])
    hi = np.array([domain[f][1] for f in FEATURES])
    return lo + (hi - lo) * rng.random((n, len(FEATURES)))


def char_func(u, tau, r, v0, kappa, theta, xi, rho):
    """E[exp(iu ln(S_T/S_0))]. Formen från Albrecher m.fl. (2007) som undviker grenskärningsproblemet."""
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b * b + xi**2 * (iu + u * u))
    g = (b - d) / (b + d)
    e = np.exp(-d * tau)
    C = r * iu * tau + kappa * theta / xi**2 * ((b - d) * tau - 2 * np.log((1 - g * e) / (1 - g)))
    D = (b - d) / xi**2 * (1 - e) / (1 - g * e)
    return np.exp(C + D * v0)


def cumulants(tau, r, v0, kappa, theta, xi, rho):
    """
    Medelvärde c1 och varians c2 av ln(S_T/S_0). c2 tas numeriskt ur den karakteristiska
    funktionen: log phi(h) = i c1 h - c2 h^2/2 + O(h^3)  =>  c2 = -2 Re log phi(h) / h^2.
    (Formeln för c2 i tabell 11 hos Fang & Oosterlee gav fel värden för stora xi när jag testade.)
    """
    ekt = np.exp(-kappa * tau)
    c1 = r * tau + (1 - ekt) * (theta - v0) / (2 * kappa) - 0.5 * theta * tau
    h = 1e-2
    c2 = -2 * np.log(char_func(h, tau, r, v0, kappa, theta, xi, rho)).real / h**2
    return c1, c2


def _cos_put_chunk(m, tau, r, v0, kappa, theta, xi, rho, N, L):
    x = np.log(m)
    c1, c2 = cumulants(tau, r, v0, kappa, theta, xi, rho)
    # Intervall för y = ln(S_T/K). Vänstersvansen är tjock när rho < 0 och xi är stor,
    # så intervallet görs asymmetriskt (L = (vänster, höger) i standardavvikelser).
    a = x + c1 - L[0] * np.sqrt(c2)
    b = x + c1 + L[1] * np.sqrt(c2)
    u = np.arange(N)[None, :] * np.pi / (b - a)[:, None]
    col = lambda z: z[:, None]
    phi = char_func(u, col(tau), col(r), col(v0), col(kappa), col(theta), col(xi), col(rho))

    # put-utbetalning (1 - e^y)^+ på [a, min(0, b)]
    d = col(np.minimum(0.0, b))
    A = col(a)
    chi = (np.cos(u * (d - A)) * np.exp(d) - np.exp(A) + u * np.sin(u * (d - A)) * np.exp(d)) / (1 + u * u)
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = np.where(u > 0, np.sin(u * (d - A)) / u, d - A)
    U = 2 / col(b - a) * (psi - chi)
    U[a >= 0] = 0.0                        # hela fördelningen över strike: puten är värdelös

    terms = phi * np.exp(1j * u * col(x - a))
    terms[:, 0] *= 0.5
    disc = np.exp(-r * tau)
    put = disc * np.sum(terms.real * U, axis=1)
    dput_dx = disc * np.sum((terms * 1j * u).real * U, axis=1)    # med fixt intervall [a, b]
    return put, dput_dx / m


def cos_call(m, tau, r, v0, kappa, theta, xi, rho, N=1024, L=(32, 12), chunk=1024, with_delta=False):
    """
    Normaliserat callpris c = C/K (och dc/dm). Callen räknas via put-call-paritet
    eftersom put-utbetalningen är begränsad, vilket gör COS stabilare.
    """
    args = np.broadcast_arrays(*[np.atleast_1d(np.asarray(z, float)) for z in
                                 (m, tau, r, v0, kappa, theta, xi, rho)])
    n = args[0].size
    args = [z.ravel() for z in args]
    put, dput = np.empty(n), np.empty(n)
    for i in range(0, n, chunk):
        sl = slice(i, i + chunk)
        put[sl], dput[sl] = _cos_put_chunk(*[z[sl] for z in args], N, L)
    m_, tau_, r_ = args[0], args[1], args[2]
    call = put + m_ - np.exp(-r_ * tau_)
    return (call, dput + 1.0) if with_delta else call


def quad_call(m, tau, r, v0, kappa, theta, xi, rho):
    """
    Gil-Pelaez: c = m P1 - e^{-r tau} P2, där P1, P2 är sannolikheter som fås genom att
    invertera den karakteristiska funktionen numeriskt. Skalär.
    """
    x = np.log(m)
    p = (tau, r, v0, kappa, theta, xi, rho)
    f1 = lambda u: (np.exp(1j * u * x) * char_func(u - 1j, *p) * np.exp(-r * tau) / (1j * u)).real
    f2 = lambda u: (np.exp(1j * u * x) * char_func(u, *p) / (1j * u)).real
    opts = dict(limit=2000, epsabs=1e-13, epsrel=1e-11)
    P1 = 0.5 + quad(f1, 0, np.inf, **opts)[0] / np.pi
    P2 = 0.5 + quad(f2, 0, np.inf, **opts)[0] / np.pi
    return m * P1 - np.exp(-r * tau) * P2


def mc_call(m, tau, r, v0, kappa, theta, xi, rho, n_paths, n_steps, rng):
    """Euler med full truncation för v och log-Euler för S, antitetiska banor. (skattning, std.fel)"""
    h = n_paths // 2
    dt = tau / n_steps
    lnS = np.full(2 * h, np.log(m))
    v = np.full(2 * h, v0)
    for _ in range(n_steps):
        z1 = rng.standard_normal(h)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(h)
        z1, z2 = np.concatenate([z1, -z1]), np.concatenate([z2, -z2])
        vp = np.maximum(v, 0.0)
        lnS += (r - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1
        v += kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * z2
    pay = np.exp(-r * tau) * np.maximum(np.exp(lnS) - 1.0, 0.0)
    pairs = 0.5 * (pay[:h] + pay[h:])
    return pairs.mean(), pairs.std(ddof=1) / np.sqrt(h)


def pde_residual(c, c_m, c_mm, c_v, c_vv, c_mv, c_tau, m, r, v, kappa, theta, xi, rho):
    """
    Heston-PDE:n i normaliserade variabler (v = nuvarande varians = v0):
    -c_tau + 1/2 v m^2 c_mm + rho xi v m c_mv + 1/2 xi^2 v c_vv + r m c_m + kappa(theta - v) c_v - r c = 0
    """
    return (-c_tau + 0.5 * v * m**2 * c_mm + rho * xi * v * m * c_mv + 0.5 * xi**2 * v * c_vv
            + r * m * c_m + kappa * (theta - v) * c_v - r * c)
