"""MLP som approximerar normaliserat optionspris c = C/K."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from black_scholes import DOMAIN


class PricerMLP(nn.Module):
    # SiLU istället för ReLU, annars blir andraderivatan (gamma) noll nästan överallt.
    # domain = dict med (min, max) per indata, används för att skala indata till [-1, 1].
    # out_act="softplus" gör utgången positiv: softplus(beta*y)/beta. Med stort beta är den nästan
    # identiteten utom nära 0, så gradienterna dämpas inte för små priser.
    def __init__(self, width=128, depth=4, domain=None, out_act="none", beta=1.0):
        super().__init__()
        domain = domain or DOMAIN
        self.domain = {k: tuple(v) for k, v in domain.items()}
        self.register_buffer("lo", torch.tensor([v[0] for v in domain.values()]))
        self.register_buffer("hi", torch.tensor([v[1] for v in domain.values()]))
        layers, d_in = [], len(domain)
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.SiLU()]
            d_in = width
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)
        self.width, self.depth, self.out_act, self.beta = width, depth, out_act, beta

    def forward(self, x):
        z = 2 * (x - self.lo) / (self.hi - self.lo) - 1
        y = self.net(z).squeeze(-1)
        return F.softplus(y, beta=self.beta) if self.out_act == "softplus" else y


def bs_call_torch(m, tau, r, sigma):
    s = sigma * torch.sqrt(tau)
    d1 = (torch.log(m) + (r + 0.5 * sigma**2) * tau) / s
    return m * torch.special.ndtr(d1) - torch.exp(-r * tau) * torch.special.ndtr(d1 - s)


class IVNet(nn.Module):
    # Nätet ger implicit volatilitet och priset räknas med Black-Scholes-formeln. Ett pris från
    # formeln med sigma > 0 ligger alltid mellan arbitragegränserna. Kolumn 0-2 måste vara m, tau, r.
    def __init__(self, width=128, depth=4, domain=None, beta=20.0):
        super().__init__()
        self.mlp = PricerMLP(width, depth, domain, out_act="softplus", beta=beta)
        self.width, self.depth, self.domain, self.beta = width, depth, self.mlp.domain, beta

    def iv(self, x):
        return self.mlp(x)

    def forward(self, x):
        return bs_call_torch(x[:, 0], x[:, 1], x[:, 2], self.iv(x))


def load_model(path):
    ckpt = torch.load(path, weights_only=False)
    if ckpt.get("kind") == "iv":
        model = IVNet(ckpt["width"], ckpt["depth"], ckpt.get("domain"), ckpt.get("beta", 20.0))
    else:
        model = PricerMLP(ckpt["width"], ckpt["depth"], ckpt.get("domain"), ckpt.get("out_act", "none"),
                          ckpt.get("beta", 1.0))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def nn_price(model, X, batch=200_000):
    model = model.double()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(torch.from_numpy(X[i:i + batch])).numpy())
    return np.concatenate(out)


def nn_derivatives(model, X, pairs=((0, 0),), batch=50_000):
    """Pris, gradient (n, d) och andraderivator d2c/dx_i dx_j för (i, j) i pairs (autograd)."""
    model = model.double()
    cs, gs, hs = [], [], {p: [] for p in pairs}
    for i in range(0, len(X), batch):
        x = torch.from_numpy(X[i:i + batch]).requires_grad_(True)
        c = model(x)
        g = torch.autograd.grad(c.sum(), x, create_graph=True)[0]
        for a, b in pairs:
            h = torch.autograd.grad(g[:, a].sum(), x, retain_graph=True)[0][:, b]
            hs[(a, b)].append(h.detach().numpy())
        cs.append(c.detach().numpy())
        gs.append(g.detach().numpy())
    return np.concatenate(cs), np.concatenate(gs), {p: np.concatenate(v) for p, v in hs.items()}


def nn_price_and_derivatives(model, X, batch=50_000):
    # Black-Scholes-nätet, kolumner (m, tau, r, sigma)
    c, g, h = nn_derivatives(model, X, ((0, 0),), batch)
    return {"c": c, "delta": g[:, 0], "gamma": h[(0, 0)], "dtau": g[:, 1], "rho": g[:, 2], "vega": g[:, 3]}
