#include "curve.h"
#include <cmath>

using namespace std;


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

double linearLogInterpolation(
    vector<pair<int,double>>& curve,
    int targetDay)
{
    for(auto& p : curve)
        if(p.first == targetDay)
            return p.second;

    for(int i = 0; i < (int)curve.size()-1; i++)
    {
        int x1 = curve[i].first;
        int x2 = curve[i+1].first;

        if(targetDay >= x1 && targetDay <= x2)
        {
            double logDF1 = log(curve[i].second);
            double logDF2 = log(curve[i+1].second);

            double w =
                (double)(targetDay - x1)
                / (x2 - x1);

            return exp(
                logDF1 +
                w * (logDF2 - logDF1)
            );
        }
    }

    return -1.0;
}

double quadraticValue(
    double x,
    double x1, double y1,
    double x2, double y2,
    double x3, double y3)
{
    double L1 =
        ((x-x2)*(x-x3))
        /
        ((x1-x2)*(x1-x3));

    double L2 =
        ((x-x1)*(x-x3))
        /
        ((x2-x1)*(x2-x3));

    double L3 =
        ((x-x1)*(x-x2))
        /
        ((x3-x1)*(x3-x2));

    return y1*L1 + y2*L2 + y3*L3;
}

double averagedQuadraticInterpolation(
    vector<pair<int,double>>& curve,
    int targetDay)
{
    for(auto& p : curve)
        if(p.first == targetDay)
            return p.second;

    int n = curve.size();

    for(int i = 0; i < n-1; i++)
    {
        int left = curve[i].first;
        int right = curve[i+1].first;

        if(targetDay >= left &&
           targetDay <= right)
        {
            if(i == 0)
                return linearLogInterpolation(
                    curve,
                    targetDay
                );

            double w1 =
                (double)(right-targetDay)
                /
                (right-left);

            double w2 =
                (double)(targetDay-left)
                /
                (right-left);

            double q1 =
                quadraticValue(
                    targetDay,
                    curve[i-1].first,
                    log(curve[i-1].second),
                    curve[i].first,
                    log(curve[i].second),
                    curve[i+1].first,
                    log(curve[i+1].second)
                );

            if(i+2 >= n)
                return exp(q1);

            double q2 =
                quadraticValue(
                    targetDay,
                    curve[i].first,
                    log(curve[i].second),
                    curve[i+1].first,
                    log(curve[i+1].second),
                    curve[i+2].first,
                    log(curve[i+2].second)
                );

            return exp(
                w1*q1 +
                w2*q2
            );
        }
    }

    return -1.0;
}

vector<pair<int,double>> buildSwapCurveLinear(
    vector<Node>& marketData)
{
    vector<pair<int,double>> curve;

    for(auto& x : marketData)
    {
        int T =
            maturityToDays(
                x.maturity
            );

        double S =
            x.swapRate / 100.0;

        if(T <= 180)
        {
            double delta =
                T / 360.0;

            curve.push_back(
                {
                    T,
                    1.0 /
                    (1.0 + S*delta)
                }
            );

            continue;
        }

        vector<int> payDates;

        for(int t=180;t<T;t+=180)
            payDates.push_back(t);

        payDates.push_back(T);

        double annuitySum = 0.0;

        for(int i=0;
            i<(int)payDates.size()-1;
            i++)
        {
            int t_prev =
                (i==0)
                ? 0
                : payDates[i-1];

            int t_curr =
                payDates[i];

            double delta =
                (t_curr-t_prev)
                /360.0;

            double df_i =
                linearLogInterpolation(
                    curve,
                    t_curr
                );

            annuitySum +=
                delta*df_i;
        }

        int t_prev_last =
            payDates[
                payDates.size()-2
            ];

        double delta_n =
            (T-t_prev_last)
            /360.0;

        double df_T =
            (1.0 -
             S*annuitySum)
            /
            (1.0 +
             S*delta_n);

        curve.push_back(
            {T,df_T}
        );
    }

    return curve;
}

vector<pair<int,double>> buildSwapCurveAQ(
    vector<Node>& marketData)
{
    vector<pair<int,double>> curve;

    for(auto& x : marketData)
    {
        int T =
            maturityToDays(
                x.maturity
            );

        double S =
            x.swapRate / 100.0;

        if(T <= 180)
        {
            double delta =
                T / 360.0;

            curve.push_back(
                {
                    T,
                    1.0 /
                    (1.0 + S*delta)
                }
            );

            continue;
        }

        vector<int> payDates;

        for(int t=180;t<T;t+=180)
            payDates.push_back(t);

        payDates.push_back(T);

        double annuitySum = 0.0;

        for(int i=0;
            i<(int)payDates.size()-1;
            i++)
        {
            int t_prev =
                (i==0)
                ? 0
                : payDates[i-1];

            int t_curr =
                payDates[i];

            double delta =
                (t_curr-t_prev)
                /360.0;

            double df_i =
                averagedQuadraticInterpolation(
                    curve,
                    t_curr
                );

            annuitySum +=
                delta*df_i;
        }

        int t_prev_last =
            payDates[
                payDates.size()-2
            ];

        double delta_n =
            (T-t_prev_last)
            /360.0;

        double df_T =
            (1.0 -
             S*annuitySum)
            /
            (1.0 +
             S*delta_n);

        curve.push_back(
            {T,df_T}
        );
    }

    return curve;
}