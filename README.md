# Neural-network-pricing-model

A neural network trained to approximate the price function of European call options, first in the
Black-Scholes model (compared with the exact formula) and then in the Heston model (compared with
the COS method). The error is measured against the reference solution and the results are verified
mathematically: Greeks, PDE residual, no-arbitrage conditions, an error bound from the maximum
principle, and calibration to volatility surfaces.

## Main results

- Black-Scholes: mean absolute error 0.85·10⁻⁴ of the strike, PDE residual 0.8 % of the size of
  its terms, no-arbitrage conditions violated in 0.27 % of test points on average.
- Heston: the network learns implied volatility instead of price, which guarantees the price bounds.
  Mean implied-volatility error 0.022 vol points, no arbitrage violations in 20 000 test points,
  about 600 times faster than the COS reference.
- Calibrating Heston to noisy volatility surfaces gives the same parameters with the network as
  with COS, about 24 times faster.

The full description, derivations and results (in Swedish) are in
[nn_prissattare/README.md](nn_prissattare/README.md).

## Getting started

    cd nn_prissattare
    pip install -r requirements.txt
    streamlit run app.py
