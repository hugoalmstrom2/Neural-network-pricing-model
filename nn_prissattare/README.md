# Neuralt nät som optionsprissättare

Ett neuralt nät tränas att approximera prisfunktionen för europeiska köpoptioner, först i
Black-Scholes, där det finns en exakt formel att jämföra med, och sedan i Heston, där det inte finns
någon sluten formel och nätet är användbart eftersom det är snabbt. I båda fallen kontrolleras först
referenslösningen, sedan mäts nätets fel mot den, och till sist kontrolleras att nätet uppfyller de
matematiska villkor som ett optionspris måste uppfylla.

## Sammanfattning

| | Black-Scholes, bästa nätet | Heston, IV-nätet |
|---|---:|---:|
| referens | exakt formel | COS-metoden, kontrollerad mot tre andra metoder |
| medelfel per option (K = 100 kr) | 0,85 öre | 0,62 öre |
| största fel på testpunkterna | 8,7 öre | 19 öre |
| PDE-residual / termernas storlek | 0,8 % | 0,9 % |
| arbitragebrott, andel punkter | 0,27 % i medel | 0 % |
| tid per option | formeln är snabbare | 1,1 µs (COS: 630 µs) |

- Referenslösningarna är kontrollerade symboliskt (sympy), med Monte Carlo och med en numerisk
  PDE-lösare respektive en andra Fourier-metod.
- De viktigaste förbättringarna var att träna även på delta, att träna längre och, för Heston,
  att låta nätet lära sig implicit volatilitet i stället för pris.
- Vid kalibrering av Heston mot en volatilitetsyta ger nätet lika bra parametrar som COS när ytan har
  realistiskt brus, ungefär 24 gånger snabbare.

Starta dashboarden med `streamlit run app.py`.

## Köra

`pip install -r requirements.txt` installerar de paket och versioner som användes. Allt sparas i
`resultat/`. Tiderna gäller en vanlig laptop-CPU.

```
python verify_exact.py          # kontroll av Black-Scholes-formeln             (15 s)
python train.py                 # första BS-nätet, bara priser                 (9 min)
python train.py --sobolev 1 --wide --hard 0.3 --shape 1 --softplus --epochs 400 --out resultat/modell_bast.pt   (55 min)
python verify_nn.py --model resultat/modell_bast.pt --tag bast                 (1-2 min)
python convergence.py           # fel mot nätstorlek                           (7 min)

python verify_heston_ref.py     # kontroll av Heston-referensen                 (5 min)
python train_heston.py --epochs 400 --out resultat/heston_modell_400.pt        (40 min första gången)
python train_heston.py --iv --epochs 400                                        (35 min)
python verify_heston_nn.py --model resultat/heston_modell_400.pt --tag 400     (3 min)
python verify_heston_nn.py                                                      (3 min)
python calibrate.py             # kalibrering mot 40 syntetiska ytor           (2 min)

streamlit run app.py            # dashboard, http://localhost:8501
```

Första gången `train_heston.py` körs räknas träningsdata med COS (ca 7 min) och sparas i
`resultat/heston_data.npz`. Mellanstegen i utvecklingen (modeller, utskrifter och figurer) ligger i
`resultat/arkiv/`.

## Black-Scholes

Priset på en köpoption är

$$C = S N(d_1) - K e^{-r\tau} N(d_2), \qquad d_{1,2} = \frac{\ln(S/K) + (r \pm \sigma^2/2)\tau}{\sigma\sqrt\tau}.$$

Eftersom $C(\lambda S, \lambda K) = \lambda C(S, K)$ räcker det att lära sig $c = C/K$ som funktion
av $m = S/K$, löptiden $\tau$, räntan $r$ och volatiliteten $\sigma$. Domänen är
$m \in [0.6, 1.4]$, $\tau \in [0.1, 2]$, $r \in [0, 0.08]$, $\sigma \in [0.1, 0.5]$.

Nätet har 4 dolda lager med 128 neuroner (ca 50 000 parametrar) och SiLU som aktivering. Med ReLU
hade andraderivatan varit noll nästan överallt och då går det inte att kontrollera gamma eller PDE:n.
Felen anges i öre per option med $K = 100$ kr, dvs. $10^4$ gånger felet i $c$.

### Kontroll av formeln

`verify_exact.py` kontrollerar formeln på tre sätt. Med sympy förenklas PDE:n, put-call-pariteten,
slutvillkoret och alla fem grekformler till exakt 0. Monte Carlo med 4 miljoner banor i tio punkter
ger $\chi^2 = 4.3$ med 10 frihetsgrader ($p = 0.93$); för optioner långt ut ur pengarna behövdes
importance sampling, annars hamnade nästan inga banor över strike. En Crank-Nicolson-lösning av
PDE:n konvergerar mot formeln med ordning 2,00.

### Från första till bästa nätet

Varianterna testades på $10^6$ nya punkter. Varje kolumn bygger på den förra.

| | bara priser | pris + delta | + riktad sampling | + 400 epoker | + straff, softplus |
|---|---:|---:|---:|---:|---:|
| medelfel (öre) | 1,94 | 1,35 | 1,76 | 0,89 | 0,85 |
| största fel (öre) | 41,3 | 25,0 | 15,8 | 10,2 | 8,7 |
| gamma, relativt fel | 4,3 % | 1,4 % | 1,3 % | 0,7 % | 0,7 % |
| PDE-residual / termernas storlek | 3,1 % | 1,9 % | 2,1 % | 1,1 % | 0,8 % |
| arbitragebrott, medel över villkoren | 2,1 % | 1,7 % | 1,5 % | 1,2 % | 0,27 % |

- Pris + delta: förlusten innehåller även felet i $\partial c/\partial m$ (Sobolev-träning).
  Det förbättrade alla derivator, fast bara delta tränades.
- Riktad sampling: felkartan visade att det största felet låg vid kort löptid och låg volatilitet
  nära ATM, där priset nästan har kinken från utbetalningen, och vid domänens kanter. 30 % av
  träningspunkterna läggs där och nätet tränas på en något större domän. Största felet minskade
  men medelfelet blev sämre, eftersom nätet lade kapacitet på de svåra områdena.
- 400 epoker: längre träning gjorde allt bättre.
- Straff och softplus: förlusten får ett straff för brott mot arbitragevillkoren (se nedan),
  räknat i 1024 nya slumpade punkter i varje steg, och utgången blir
  $c = \text{softplus}(\beta y)/\beta$ med $\beta = 100$ så att priset inte kan bli negativt.
  Vanlig softplus ($\beta = 1$) lärde sig för långsamt, eftersom lutningen då är ungefär lika med
  priset och gradienterna dämpas för billiga optioner. Eftersom den sanna funktionen uppfyller
  villkoren kostade straffet ingen noggrannhet.

Testfelet och träningsfelet är ungefär lika stora i alla varianter, så näten är inte överanpassade.
Konvergensstudien (`fig_konvergens.png`) visar att felet minskar när nätet görs bredare.

### Matematisk kontroll av nätet

`verify_nn.py` gör följande.

Greker räknas genom att derivera nätet med autograd och jämförs med formlerna.

PDE-residualen $R = -c_\tau + \tfrac12\sigma^2 m^2 c_{mm} + r m c_m - r c$ ska vara 0. Formeln
ger $10^{-16}$ och det bästa nätet 0,8 % av termernas storlek. Nätet har aldrig sett PDE:n, så det
är en oberoende kontroll.

Arbitragevillkoren $\max(m - e^{-r\tau}, 0) \le c \le m$, $0 \le \Delta \le 1$, $\Gamma \ge 0$,
vega $\ge 0$, $c_\tau \ge 0$ och $\rho \ge 0$. Utan straff bröts de i 0,6-2,8 % av punkterna, alltid
med mindre än nätets eget fel i samma storhet, där det sanna värdet ligger på gränsen. Det bästa nätet
bryter dem i 0,27 % av punkterna i medel. Det som är kvar är främst den nedre gränsen för optioner
djupt in the money (1,0 %), där gränsen är $m - e^{-r\tau} > 0$ och softplus inte hjälper.

Felgräns från maximumprincipen. Felet $e = c_{NN} - c$ uppfyller $e_\tau = \mathcal{L}e - R$ med
$\mathcal{L} = \tfrac12\sigma^2m^2\partial_{mm} + rm\partial_m - r$. Sätt
$B_0 = \max |e|$ på randen $\{\tau = \tau_{\min}\} \cup \{m = m_{\min}\} \cup \{m = m_{\max}\}$ och
$v = B_0 + (\tau - \tau_{\min})\max|R|$. Då är $(e - v)_\tau - \mathcal{L}(e - v) = -R - \max|R| - rv \le 0$
och $e - v \le 0$ på randen, så den svaga maximumprincipen ger $e \le v$. Samma sak för $-e$ ger

$$|e(m, \tau)| \le \max_{\text{rand}} |e| + (\tau - \tau_{\min}) \max |R|.$$

Gränsen behöver bara felet på randen och nätets egen residual, inte den exakta lösningen inne i
området. Den håller i alla 15 testade kombinationer av $(r, \sigma)$ för alla nät.

Statistisk garanti. Med $N = 10^6$ testpunkter och inget fel större än $\varepsilon$ är andelen
av domänen med fel över $\varepsilon$ mindre än $-\ln(0.05)/N \approx 3 \cdot 10^{-6}$ med 95 %
konfidens, eftersom $(1-p)^N \le 0.05$ annars.

## Heston

I Heston är variansen stokastisk:

$$dS = rS\,dt + \sqrt{v}\,S\,dW_1, \qquad dv = \kappa(\theta - v)\,dt + \xi\sqrt{v}\,dW_2, \qquad d\langle W_1, W_2\rangle = \rho\,dt.$$

Nätet har 8 indata: $m, \tau, r, v_0, \kappa, \theta, \xi, \rho$ med
$v_0, \theta \in [0.01, 0.25]$, $\kappa \in [0.5, 5]$, $\xi \in [0.1, 1]$ och $\rho \in [-0.9, 0]$.
Samma nätstorlek som för BS, tränat i 400 epoker med pris och delta och samma riktade sampling.

### Referenslösning

Det finns ingen sluten formel, men den karakteristiska funktionen för $\ln S_T$ är känd. Referensen
är COS-metoden (Fang & Oosterlee 2008), som utvecklar täthetsfunktionen i en cosinusserie.
`verify_heston_ref.py` kontrollerar den:

- Testfallet från Fang & Oosterlee (5,785155450) återges på $1.6 \cdot 10^{-8}$ när.
- En annan Fourier-metod (Gil-Pelaez-integralen med `scipy.quad`) skiljer sig högst
  $7.9 \cdot 10^{-7}$, och i median $4 \cdot 10^{-16}$, på 364 punkter inklusive domänens hörn.
- När $\xi \to 0$ ska priset bli Black-Scholes med $\sigma_{\text{eff}}^2 = \frac1\tau\int_0^\tau v(t)\,dt$.
  Skillnaden går mot 0 med ordning 1 när $\rho \ne 0$ och ordning 2 när $\rho = 0$, som teorin säger.
- Monte Carlo stämmer inom 2,4 standardfel i fem punkter. Med för få tidssteg syns Euler-schemats
  bias ($z = 4.7$ med 125 steg per år); med 500 steg eller fler försvinner den.
- COS-priset uppfyller Heston-PDE:n med residual ca $4 \cdot 10^{-7}$.

Två saker fick rättas på vägen. Formeln för andra kumulanten i artikelns tabell 11 gav fel värden för
stora $\xi$, så den räknas nu numeriskt ur den karakteristiska funktionen. Och integrationsintervallet
måste vara asymmetriskt (32 standardavvikelser åt vänster, 12 åt höger), eftersom vänstersvansen är
tjock när $\rho < 0$ och $\xi$ är stort.

### Pris-nät och IV-nät

Det första Heston-nätet lärde sig priset direkt. Dess svaga punkter var optioner med lite tidsvärde,
där felet i implicit vol drog iväg, och brott mot prisgränserna. Båda beror på att priset där är
nästan 0. IV-nätet (`IVNet` i `nn_model.py`) ger i stället implicit volatilitet $\sigma_{NN} > 0$,
och priset räknas exakt med Black-Scholes-formeln från $\sigma_{NN}$. Ett sådant pris ligger alltid
mellan $\max(m - e^{-r\tau}, 0)$ och $m$, så prisgränserna gäller per konstruktion. Förlusten är
felet i volatilitet plus felet i delta. Samma COS-data användes; 1 % av punkterna togs bort eftersom
vega var under $10^{-4}$ och volatiliteten då inte går att bestämma säkert.

På $10^5$ testpunkter:

| | pris-nät | IV-nät |
|---|---:|---:|
| medelfel (öre) | 2,4 | 0,62 |
| största fel (öre) | 31 | 19 |
| implicit vol, medel / 99 % / max (volpunkter) | 0,085 / 0,46 / 1,8 | 0,022 / 0,11 / 0,73 |
| PDE-residual / termernas storlek | 3,5 % | 0,9 % |
| arbitragebrott, andel punkter | 0,5-1,2 % | 0 % för alla sex villkor |

Implicit vol räknas för optioner med tidsvärde minst 0,5 kr. IV-nätet blev bättre på alla mått.
Prisgränserna var garanterade, men även konvexitet, $c_\tau \ge 0$ och $\partial c/\partial v_0 \ge 0$
höll i alla 20 000 testpunkter fast inget av det tvingades fram. Leendefiguren
(`fig_heston_leende_iv.png`) följer referensen nästan exakt; störst avvikelse är ca 0,7
volatilitetspunkter för höga strikes vid hög vol-of-vol och kort löptid.

Tid per option på CPU:

| metod | tid |
|---|---:|
| nätet | 1,1 µs |
| COS, N = 128 | 59 µs |
| COS, N = 1024 (referensen) | 630 µs |
| quad (Gil-Pelaez) | 7 ms |
| Monte Carlo, 100 000 banor | 0,6 s |

### Kalibrering

`calibrate.py` testar om nätet duger till det det är tänkt för. En volatilitetsyta med 9 strikes
(K/S = 0,8-1,2) och 5 löptider (0,25-2 år) räknas fram med COS från kända parametrar. Sedan söks de
parametrar $(v_0, \kappa, \theta, \xi, \rho)$ som minimerar de vega-viktade prisfelen (ungefär felen i
implicit vol) med `scipy.optimize.least_squares`, en gång med IV-nätet och en gång med COS. Det gjordes
för 40 slumpade parameteruppsättningar, utan brus och med 0,25 volatilitetspunkters brus på ytan
(ungefär ett bid-ask-spread).

Median av absolutfelet i parametrarna:

| | $v_0$ | $\kappa$ | $\theta$ | $\xi$ | $\rho$ | passning (volp.) |
|---|---:|---:|---:|---:|---:|---:|
| nät, utan brus | 0,0003 | 0,026 | 0,0001 | 0,006 | 0,004 | 0,016 |
| COS, utan brus | 0 | 0 | 0 | 0 | 0 | 0 |
| nät, med brus | 0,0012 | 0,15 | 0,0010 | 0,040 | 0,034 | 0,234 |
| COS, med brus | 0,0013 | 0,17 | 0,0010 | 0,040 | 0,036 | 0,233 |

"Passning" är RMSE i implicit vol när de kalibrerade parametrarna prissätts med COS; de sanna
parametrarna ger 0,25 på ytorna med brus. Utan brus hittar COS exakt rätt, vilket visar att
kalibreringen fungerar, och nätets fel ger ett litet systematiskt fel i parametrarna. Med realistiskt
brus går nätet och COS inte att skilja åt. $\kappa$ är svårast för båda; med löptider upp till 2 år
syns återgångshastigheten bara svagt i ytan.

Nätets Jacobian mot parametrarna räknas med autograd i ett enda bakåtpass (samma som finita
differenser på $8 \cdot 10^{-11}$ när). Kalibreringen går ungefär 24 gånger snabbare med nätet. Det är
mycket mindre än 600 gånger, eftersom de fasta kostnaderna per anrop och optimeraren dominerar när
ytan bara har 45 optioner.

## Dashboard

`streamlit run app.py` startar en lokal sida med fyra flikar:

- Black-Scholes: välj nät och parametrar och se pris, fel, greker eller PDE-residual längs S/K
  och en felkarta. Det går att gå utanför träningsdomänen för att se hur nätet extrapolerar.
- Heston: välj pris- eller IV-nätet, ställ in modellparametrarna och jämför volatilitetsleendet
  med COS.
- Kalibrering: välj sanna parametrar och brus och se hur nätet och COS kalibrerar mot ytan.
- Verifiering: de sammanfattade resultaten.

## Begränsningar

- Utanför träningsdomänen blir felen snabbt stora.
- För Black-Scholes garanteras inte den nedre prisgränsen djupt in the money. IV-knepet fungerar inte
  där, eftersom volatiliteten redan är en av indata.
- Kalibreringen är bara testad på syntetiska ytor. Nästa steg vore en riktig optionskedja, t.ex.
  europeiska indexoptioner (jfr Horvath, Muguruza och Tomas, "Deep learning volatility", 2019).

## Filer

- `black_scholes.py`, `heston.py`: prissättning (formel, COS, quad, Monte Carlo, Crank-Nicolson)
- `nn_model.py`, `train.py`, `train_heston.py`: näten och träningen
- `verify_exact.py`, `verify_heston_ref.py`: kontroll av referenslösningarna
- `verify_nn.py`, `verify_heston_nn.py`, `convergence.py`: nätens fel och matematiska kontroller
- `calibrate.py`: kalibrering av Heston
- `app.py`, `plot_style.py`: dashboard och figurstil
- `resultat/`: slutliga modeller, utskrifter (`.txt`), sammanfattningar (`.json`) och figurer
- `resultat/arkiv/`: mellanstegen
