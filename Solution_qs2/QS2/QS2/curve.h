#ifndef CURVE_H
#define CURVE_H

#include <vector>
#include <string>

using namespace std;

struct Node
{
    string maturity;
    double cashRate;
    double swapRate;
};

int maturityToDays(string s);

double cashDF(
    double ratePercent,
    int days
);

double linearLogInterpolation(
    vector<pair<int,double>>& curve,
    int targetDay
);

double averagedQuadraticInterpolation(
    vector<pair<int,double>>& curve,
    int targetDay
);

vector<pair<int,double>> buildSwapCurveLinear(
    vector<Node>& marketData
);

vector<pair<int,double>> buildSwapCurveAQ(
    vector<Node>& marketData
);

#endif