# Quant Challenge and what was done (and future improvements)
Adversarial client profiling - building a framework to profile/classify client behavior in an adversarial trading context

XGBoost toxic flow classifier - achieving 55.6% test accuracy using chronological train/test splits (not random splits, which matters for avoiding lookahead bias in financial data)

Dynamic quoting engine - a market-making style quoting strategy that achieved a Sharpe ratio of ~15 in backtest(Ik its a lot so working on figuring out where i might have overfit the data / or maybe try using DL , lets see)

Externalization threshold framework - a framework for deciding when to externalize (offload to the broader market) versus internalize order flow

C++ interest rate curve construction engine - building curve bootstrapping, DCF valuation, IRS (interest rate swap) pricing, and both linear and quadratic interpolation methods (pretty basic but a good Object Oriented Programming practise , could be evaluated further for any arbitary instrument as input which is what i will try to do )











# Interest Rate Curve & Swap Pricer (Question 2)

Build and run the C++ example that constructs discount curves from cash and swap instruments,
interpolates discount factors, and prices a sample swap.

Compile:

```bash
g++ -std=c++17 question2.cpp -O2 -o question2
```

Run (uses sample CSVs in `data/`):

```bash
./question2
```

Files:
- `question2.cpp` — main program.
- `data/cash_rates.csv` — sample cash deposit rates (days, rate).
- `data/swap_rates.csv` — sample par swap rates (days, rate).
- `data/new_swap.csv` — new swap to price (maturity_days, fixedRate).

Notes:
- Interpolation: Linear and Averaged Quadratic (average of two adjacent quadratics when available).
- Cash DF convention: simple deposit DF = 1/(1 + r * T).
- Swap bootstrap: annual payments, year fraction = 1.

Adjust CSVs to match your dataset and re-run.
