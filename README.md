# Neural-network-pricing-model

Neuralt nät som approximerar prisfunktionen för europeiska köpoptioner, i Black-Scholes
(mot den exakta formeln) och i Heston (mot COS-metoden). Felet mäts mot referenslösningen och
resultatet verifieras matematiskt: greker, PDE-residual, arbitragevillkor, en felgräns från
maximumprincipen och kalibrering mot volatilitetsytor.

Beskrivning, resultat och instruktioner finns i [nn_prissattare/README.md](nn_prissattare/README.md).

Kom igång:

    cd nn_prissattare
    pip install -r requirements.txt
    streamlit run app.py
