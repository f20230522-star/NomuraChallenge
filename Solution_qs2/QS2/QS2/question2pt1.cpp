#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

struct Node
{
    string maturity;
    double cashRate;
    double swapRate;
};

int maturityToDays(string s)
{
    int num = stoi(s.substr(0, s.size()-1));
    char unit = s.back();

    if(unit == 'D') return num;
    if(unit == 'W') return num * 7;
    if(unit == 'M') return num * 30;
    if(unit == 'Y') return num * 360;

    return -1;
}

double cashDF(double ratePercent, int days)
{
    double r = ratePercent / 100.0;
    return 1.0 / (1.0 + r * days / 360.0);
}

// ─── interpolation ───────────────────────────────────────────

double linearLogInterpolation(
    vector<pair<int,double>>& curve,
    int targetDay)
{
    // exact node hit
    for(auto& p : curve)
        if(p.first == targetDay)
            return p.second;

    for(int i = 0; i < (int)curve.size()-1; i++)
    {
        int    x1 = curve[i].first;
        int    x2 = curve[i+1].first;

        if(targetDay >= x1 && targetDay <= x2)
        {
            double logDF1 = log(curve[i].second);
            double logDF2 = log(curve[i+1].second);

            double w = (double)(targetDay - x1) / (x2 - x1);

            return exp(logDF1 + w * (logDF2 - logDF1));
        }
    }

    return -1.0; // out of range
}

double quadraticValue(
    double x,
    double x1, double y1,
    double x2, double y2,
    double x3, double y3)
{
    double L1 = ((x-x2)*(x-x3)) / ((x1-x2)*(x1-x3));
    double L2 = ((x-x1)*(x-x3)) / ((x2-x1)*(x2-x3));
    double L3 = ((x-x1)*(x-x2)) / ((x3-x1)*(x3-x2));

    return y1*L1 + y2*L2 + y3*L3;
}

double averagedQuadraticInterpolation(
    vector<pair<int,double>>& curve,
    int targetDay)
{
    // exact node hit
    for(auto& p : curve)
        if(p.first == targetDay)
            return p.second;

    int n = curve.size();

    for(int i = 0; i < n-1; i++)
    {
        int left  = curve[i].first;
        int right = curve[i+1].first;

        if(targetDay >= left && targetDay <= right)
        {
            // first interval: fall back to linear (no left neighbour)
            if(i == 0)
                return linearLogInterpolation(curve, targetDay);

            // need i+2 for Q2; if it doesn't exist use Q1 only
            double w1 = (double)(right - targetDay) / (right - left);
            double w2 = (double)(targetDay - left)  / (right - left);

            double q1 = quadraticValue(
                targetDay,
                curve[i-1].first, log(curve[i-1].second),
                curve[i  ].first, log(curve[i  ].second),
                curve[i+1].first, log(curve[i+1].second));

            if(i+2 >= n)
                return exp(q1); // last interval: only Q_i available

            double q2 = quadraticValue(
                targetDay,
                curve[i  ].first, log(curve[i  ].second),
                curve[i+1].first, log(curve[i+1].second),
                curve[i+2].first, log(curve[i+2].second));

            return exp(w1*q1 + w2*q2);
        }
    }

    return -1.0;
}

// ─── swap curve bootstrap ─────────────────────────────────────

vector<pair<int,double>> buildSwapCurveLinear(
    vector<Node>& marketData)
{
    vector<pair<int,double>> curve;

    for(auto& x : marketData)
    {
        int    T = maturityToDays(x.maturity);
        double S = x.swapRate / 100.0;

        // short end: single payment, identical to cash formula
        if(T <= 180)
        {
            double delta = T / 360.0;
            curve.push_back({T, 1.0 / (1.0 + S * delta)});
            continue;
        }

        // build semi-annual payment schedule
        vector<int> payDates;
        for(int t = 180; t < T; t += 180)
            payDates.push_back(t);
        payDates.push_back(T);

        // annuity sum over all but last payment date
        double annuitySum = 0.0;

        for(int i = 0; i < (int)payDates.size()-1; i++)
        {
            int t_prev = (i == 0) ? 0 : payDates[i-1];
            int t_curr = payDates[i];
            double delta = (t_curr - t_prev) / 360.0;

            double df_i = linearLogInterpolation(curve, t_curr);
            annuitySum += delta * df_i;
        }

        // last period
        int    t_prev_last = payDates[payDates.size()-2];
        double delta_n     = (T - t_prev_last) / 360.0;

        double df_T = (1.0 - S * annuitySum) / (1.0 + S * delta_n);
        curve.push_back({T, df_T});
    }

    return curve;
}

vector<pair<int,double>> buildSwapCurveAQ(
    vector<Node>& marketData)
{
    vector<pair<int,double>> curve;

    for(auto& x : marketData)
    {
        int    T = maturityToDays(x.maturity);
        double S = x.swapRate / 100.0;

        // short end
        if(T <= 180)
        {
            double delta = T / 360.0;
            curve.push_back({T, 1.0 / (1.0 + S * delta)});
            continue;
        }

        // payment schedule
        vector<int> payDates;
        for(int t = 180; t < T; t += 180)
            payDates.push_back(t);
        payDates.push_back(T);

        // annuity sum — use AQ interpolation this time
        double annuitySum = 0.0;

        for(int i = 0; i < (int)payDates.size()-1; i++)
        {
            int t_prev = (i == 0) ? 0 : payDates[i-1];
            int t_curr = payDates[i];
            double delta = (t_curr - t_prev) / 360.0;

            double df_i = averagedQuadraticInterpolation(curve, t_curr);
            annuitySum += delta * df_i;
        }

        int    t_prev_last = payDates[payDates.size()-2];
        double delta_n     = (T - t_prev_last) / 360.0;

        double df_T = (1.0 - S * annuitySum) / (1.0 + S * delta_n);
        curve.push_back({T, df_T});
    }

    return curve;
}

// ─── main ─────────────────────────────────────────────────────

int main()
{
    vector<Node> marketData =
    {
        {"1D",  3.64, 3.55},
        {"1W",  3.65, 3.58},
        {"2W",  6.75, 3.63},
        {"1M",  5.08, 3.69},
        {"2M",  5.00, 3.74},
        {"3M",  4.97, 3.79},
        {"6M",  5.96, 3.84},
        {"1Y",  6.96, 3.93},
        {"2Y",  4.95, 4.05},
        {"3Y",  7.95, 4.15},
        {"5Y",  8.95, 4.16},
        {"10Y", 4.99, 4.14},
        {"20Y", 9.96, 4.08},
        {"40Y", 4.95, 4.03}
    };

    // ── Q1a: cash curve + linear interpolation ──
    vector<pair<int,double>> cashCurve;
    for(auto& x : marketData)
    {
        int days = maturityToDays(x.maturity);
        cashCurve.push_back({days, cashDF(x.cashRate, days)});
    }

    int queryDay = 784;

    cout << "Q1a) DF(" << queryDay << ") Cash/Linear = "
         << linearLogInterpolation(cashCurve, queryDay) << "\n";

    // ── Q1b: cash curve + AQ interpolation ──
    cout << "Q1b) DF(" << queryDay << ") Cash/AQ     = "
         << averagedQuadraticInterpolation(cashCurve, queryDay) << "\n";

    // ── Q1c: swap curve + linear interpolation ──
    vector<pair<int,double>> swapLinear = buildSwapCurveLinear(marketData);

    cout << "Q1c) DF(" << queryDay << ") Swap/Linear = "
         << linearLogInterpolation(swapLinear, queryDay) << "\n";

    // ── Q1d: swap curve + AQ interpolation ──
    vector<pair<int,double>> swapAQ = buildSwapCurveAQ(marketData);

    cout << "Q1d) DF(" << queryDay << ") Swap/AQ     = "
         << averagedQuadraticInterpolation(swapAQ, queryDay) << "\n";

    return 0;
}