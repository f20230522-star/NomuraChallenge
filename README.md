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
