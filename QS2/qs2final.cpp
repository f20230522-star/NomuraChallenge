#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <fstream>
#include <iomanip>

using namespace std;

struct Node {
    string maturity;
    double cashRate;
    double swapRate;
};

int maturityToDays(string s) {
    int num = stoi(s.substr(0, s.size()-1));
    char unit = s.back();
    if(unit == 'D') return num;
    if(unit == 'W') return num * 7;
    if(unit == 'M') return num * 30;
    if(unit == 'Y') return num * 360;
    return -1;
}

double cashDF(double ratePercent, int days) {
    double r = ratePercent / 100.0;
    return 1.0 / (1.0 + r * days / 360.0);
}

double linearLogInterpolation(vector<pair<int,double>>& curve, int t) {
    for(auto& p : curve)
        if(p.first == t) return p.second;
    for(int i = 0; i < (int)curve.size()-1; i++) {
        int left = curve[i].first, right = curve[i+1].first;
        if(t >= left && t <= right) {
            double w = (double)(t-left)/(right-left);
            return exp(log(curve[i].second)*(1-w) + log(curve[i+1].second)*w);
        }
    }
    return -1.0;
}

double dLinear_dNode(vector<pair<int,double>>& curve, int t, int k) {
    if(curve[k].first == t) return 1.0;
    for(int i = 0; i < (int)curve.size()-1; i++) {
        int left = curve[i].first, right = curve[i+1].first;
        if(t > left && t < right) {
            double w = (double)(t-left)/(right-left);
            double df_t = linearLogInterpolation(curve, t);
            if(k == i)   return df_t*(1.0-w)/curve[i].second;
            if(k == i+1) return df_t*w/curve[i+1].second;
            return 0.0;
        }
    }
    return 0.0;
}

static double quadraticValue(double x,
    double x1, double y1, double x2, double y2, double x3, double y3) {
    double L1 = ((x-x2)*(x-x3))/((x1-x2)*(x1-x3));
    double L2 = ((x-x1)*(x-x3))/((x2-x1)*(x2-x3));
    double L3 = ((x-x1)*(x-x2))/((x3-x1)*(x3-x2));
    return y1*L1 + y2*L2 + y3*L3;
}

double averagedQuadraticInterpolation(vector<pair<int,double>>& curve, int t) {
    for(auto& p : curve)
        if(p.first == t) return p.second;
    int n = curve.size();
    for(int i = 0; i < n-1; i++) {
        int left = curve[i].first, right = curve[i+1].first;
        if(t >= left && t <= right) {
            if(i == 0) return linearLogInterpolation(curve, t);
            double w1 = (double)(right-t)/(right-left);
            double w2 = (double)(t-left)/(right-left);
            double q1 = quadraticValue(t,
                curve[i-1].first, log(curve[i-1].second),
                curve[i  ].first, log(curve[i  ].second),
                curve[i+1].first, log(curve[i+1].second));
            if(i+2 >= n) return exp(q1);
            double q2 = quadraticValue(t,
                curve[i  ].first, log(curve[i  ].second),
                curve[i+1].first, log(curve[i+1].second),
                curve[i+2].first, log(curve[i+2].second));
            return exp(w1*q1 + w2*q2);
        }
    }
    return -1.0;
}

double dAQ_dNode(vector<pair<int,double>>& curve, int t, int k) {
    if(curve[k].first == t) return 1.0;
    int n = curve.size();
    for(int i = 0; i < n-1; i++) {
        int left = curve[i].first, right = curve[i+1].first;
        if(t > left && t < right) {
            if(i == 0) return dLinear_dNode(curve, t, k);
            double w1 = (double)(right-t)/(right-left);
            double w2 = (double)(t-left)/(right-left);
            double df_t = averagedQuadraticInterpolation(curve, t);

            auto L = [](double x, double xa, double xb, double xc, int which) -> double {
                if(which==0) return ((x-xb)*(x-xc))/((xa-xb)*(xa-xc));
                if(which==1) return ((x-xa)*(x-xc))/((xb-xa)*(xb-xc));
                return             ((x-xa)*(x-xb))/((xc-xa)*(xc-xb));
            };

            double dlogDF_dlogDFk = 0.0;
            // Q1: triplet i-1, i, i+1
            {
                double xa=curve[i-1].first, xb=curve[i].first, xc=curve[i+1].first;
                if(k==i-1) dlogDF_dlogDFk += w1*L(t,xa,xb,xc,0);
                else if(k==i)   dlogDF_dlogDFk += w1*L(t,xa,xb,xc,1);
                else if(k==i+1) dlogDF_dlogDFk += w1*L(t,xa,xb,xc,2);
            }
            // Q2: triplet i, i+1, i+2
            if(i+2 < n) {
                double xa=curve[i].first, xb=curve[i+1].first, xc=curve[i+2].first;
                if(k==i)   dlogDF_dlogDFk += w2*L(t,xa,xb,xc,0);
                else if(k==i+1) dlogDF_dlogDFk += w2*L(t,xa,xb,xc,1);
                else if(k==i+2) dlogDF_dlogDFk += w2*L(t,xa,xb,xc,2);
            }
            return df_t * dlogDF_dlogDFk / curve[k].second;
        }
    }
    return 0.0;
}

vector<pair<int,double>> buildCashCurve(vector<Node>& md) {
    vector<pair<int,double>> curve;
    for(auto& x : md)
        curve.push_back({maturityToDays(x.maturity), cashDF(x.cashRate, maturityToDays(x.maturity))});
    return curve;
}

static vector<pair<int,double>> bootstrapSwap(vector<Node>& md, bool useAQ) {
    vector<pair<int,double>> curve;
    for(auto& x : md) {
        int T = maturityToDays(x.maturity);
        double S = x.swapRate / 100.0;
        if(T <= 180) {
            curve.push_back({T, 1.0/(1.0+S*T/360.0)});
            continue;
        }
        vector<int> pay;
        for(int t=180; t<T; t+=180) pay.push_back(t);
        pay.push_back(T);

        double ann = 0.0; int prev = 0;
        for(int i=0; i<(int)pay.size()-1; i++) {
            int tc = pay[i];
            double delta = (tc-prev)/360.0;
            double df = useAQ ? averagedQuadraticInterpolation(curve,tc)
                              : linearLogInterpolation(curve,tc);
            ann += delta*df;
            prev = tc;
        }
        int lp = pay[pay.size()-2];
        double dn = (T-lp)/360.0;
        curve.push_back({T, (1.0-S*ann)/(1.0+S*dn)});
    }
    return curve;
}

vector<pair<int,double>> buildSwapCurveLinear(vector<Node>& md) { return bootstrapSwap(md,false); }
vector<pair<int,double>> buildSwapCurveAQ(vector<Node>& md)     { return bootstrapSwap(md,true);  }

// ── swap pricing ──────────────────────────────

struct SwapSpec { double notional, fixedRate; int maturityDays, fixedFreqDays, floatFreqDays; };

vector<int> sched(int freq, int mat) {
    vector<int> d;
    for(int t=freq; t<mat; t+=freq) d.push_back(t);
    d.push_back(mat);
    return d;
}

double swapPV(const SwapSpec& sw, vector<pair<int,double>>& curve, bool useAQ) {
    auto interp = [&](int t) {
        return useAQ ? averagedQuadraticInterpolation(curve,t)
                     : linearLogInterpolation(curve,t);
    };
    double pvFloat = sw.notional*(1.0 - interp(sw.maturityDays));
    double pvFixed = 0.0; int prev=0;
    for(int t : sched(sw.fixedFreqDays, sw.maturityDays)) {
        pvFixed += sw.notional*sw.fixedRate*(t-prev)/360.0*interp(t);
        prev=t;
    }
    return pvFloat - pvFixed;
}

double parRate(const SwapSpec& sw, vector<pair<int,double>>& curve, bool useAQ) {
    auto interp = [&](int t) {
        return useAQ ? averagedQuadraticInterpolation(curve,t)
                     : linearLogInterpolation(curve,t);
    };
    double ann=0.0; int prev=0;
    for(int t : sched(sw.fixedFreqDays, sw.maturityDays)) {
        ann += interp(t)*(t-prev)/360.0; prev=t;
    }
    return (1.0-interp(sw.maturityDays))/ann;
}

// ── analytical risk ───────────────────────────

map<int,double> dPV_dDF_map(const SwapSpec& sw) {
    map<int,double> s;
    double N=sw.notional, K=sw.fixedRate;
    int prev=0;
    for(int t : sched(sw.fixedFreqDays, sw.maturityDays)) {
        s[t] -= N*K*(t-prev)/360.0; prev=t;
    }
    prev=0;
    for(int t : sched(sw.floatFreqDays, sw.maturityDays)) {
        if(prev!=0) s[prev]+=N;
        s[t]-=N; prev=t;
    }
    return s;
}

// dDF_i/dp_i for cash node i
double dDFi_cash(vector<pair<int,double>>& curve, int i) {
    double Ti=curve[i].first, DFi=curve[i].second;
    return -DFi*DFi*(Ti/360.0)/100.0;
}

// dDF_i/dp_i for swap node i
double dDFi_swap(vector<pair<int,double>>& curve, vector<Node>& md, int i, bool useAQ) {
    int Ti=curve[i].first;
    double Si=md[i].swapRate/100.0;
    if(Ti<=180) {
        double d=Ti/360.0, den=1.0+Si*d;
        return -d/(den*den)/100.0;
    }
    auto interp=[&](int t){ return useAQ?averagedQuadraticInterpolation(curve,t):linearLogInterpolation(curve,t); };
    double ann=0.0; int prev=0;
    vector<int> pay;
    for(int t=180;t<Ti;t+=180) pay.push_back(t);
    for(int t:pay){ ann+=(t-prev)/360.0*interp(t); prev=t; }
    int lp=pay.empty()?0:pay.back();
    double dn=(Ti-lp)/360.0, den=1.0+Si*dn;
    return (-ann-dn*curve[i].second)/den/100.0;
}

// forward propagation: dNode[j][i] = dDF_j/dDF_i
vector<vector<double>> buildProp(vector<pair<int,double>>& curve, vector<Node>& md, bool useAQ) {
    int n=curve.size();
    vector<vector<double>> D(n,vector<double>(n,0.0));
    for(int i=0;i<n;i++) D[i][i]=1.0;

    auto dI=[&](int t,int k){ return useAQ?dAQ_dNode(curve,t,k):dLinear_dNode(curve,t,k); };

    for(int j=1;j<n;j++) {
        int Tj=curve[j].first;
        double Sj=md[j].swapRate/100.0;
        if(Tj<=180) continue;
        vector<int> pay;
        for(int t=180;t<Tj;t+=180) pay.push_back(t);
        int lp=pay.empty()?0:pay.back();
        double dn=(Tj-lp)/360.0, den=1.0+Sj*dn;

        for(int k=0;k<j;k++) {
            double dAnn=0.0; int prev=0;
            for(int t:pay){ dAnn+=(t-prev)/360.0*dI(t,k); prev=t; }
            double direct=-Sj*dAnn/den;
            for(int m=k;m<j;m++) D[j][k]+=direct*D[m][k];
        }
    }
    return D;
}

vector<double> analyticalRisk(const SwapSpec& sw, vector<pair<int,double>>& curve,
    vector<Node>& md, bool isCash, bool useAQ)
{
    int n=curve.size();
    auto sens=dPV_dDF_map(sw);
    auto D=buildProp(curve,md,useAQ);
    auto dI=[&](int t,int k){ return useAQ?dAQ_dNode(curve,t,k):dLinear_dNode(curve,t,k); };

    vector<double> risk(n,0.0);
    for(int i=0;i<n;i++) {
        double dq = isCash ? dDFi_cash(curve,i) : dDFi_swap(curve,md,i,useAQ);
        for(auto&[t,s]:sens)
            for(int k=0;k<n;k++)
                risk[i]+=s*dI(t,k)*D[k][i]*dq;
    }
    return risk;
}

int main() {
    vector<Node> md = {
        {"1D",3.64,3.55},{"1W",3.65,3.58},{"2W",6.75,3.63},
        {"1M",5.08,3.69},{"2M",5.00,3.74},{"3M",4.97,3.79},
        {"6M",5.96,3.84},{"1Y",6.96,3.93},{"2Y",4.95,4.05},
        {"3Y",7.95,4.15},{"5Y",8.95,4.16},{"10Y",4.99,4.14},
        {"20Y",9.96,4.08},{"40Y",4.95,4.03}
    };

    auto cashCurve  = buildCashCurve(md);
    auto swapLinear = buildSwapCurveLinear(md);
    auto swapAQ     = buildSwapCurveAQ(md);

    int q = 784;
    SwapSpec sw = {100.0, 0.06, 25*360, 30, 180};

    // Q1
    double q1a=linearLogInterpolation(cashCurve,q);
    double q1b=averagedQuadraticInterpolation(cashCurve,q);
    double q1c=linearLogInterpolation(swapLinear,q);
    double q1d=averagedQuadraticInterpolation(swapAQ,q);

    // Q2.1
    double pv_cl=swapPV(sw,cashCurve,false),  par_cl=parRate(sw,cashCurve,false);
    double pv_ca=swapPV(sw,cashCurve,true),   par_ca=parRate(sw,cashCurve,true);
    double pv_sl=swapPV(sw,swapLinear,false), par_sl=parRate(sw,swapLinear,false);
    double pv_sa=swapPV(sw,swapAQ,true),      par_sa=parRate(sw,swapAQ,true);

    // Q2.2
    auto rCL=analyticalRisk(sw,cashCurve, md,true, false);
    auto rCA=analyticalRisk(sw,cashCurve, md,true, true);
    auto rSL=analyticalRisk(sw,swapLinear,md,false,false);
    auto rSA=analyticalRisk(sw,swapAQ,   md,false,true);

    // print all
    cout<<fixed<<setprecision(10);
    cout<<"Q1: "<<q1a<<","<<q1b<<","<<q1c<<","<<q1d<<"\n";
    cout<<"PV: "<<pv_cl<<","<<pv_ca<<","<<pv_sl<<","<<pv_sa<<"\n";
    cout<<"PAR:"<<par_cl*100<<","<<par_ca*100<<","<<par_sl*100<<","<<par_sa*100<<"\n";
    cout<<"RISK:\n";
    for(int i=0;i<14;i++)
        cout<<rCL[i]<<","<<rCA[i]<<","<<rSL[i]<<","<<rSA[i]<<"\n";

    // write csv
    ofstream f("output.csv");
    f<<fixed<<setprecision(10);
    f<<q1a<<","<<q1b<<","<<q1c<<","<<q1d<<"\n";
    f<<pv_cl<<","<<pv_ca<<","<<pv_sl<<","<<pv_sa<<"\n";
    f<<par_cl*100<<","<<par_ca*100<<","<<par_sl*100<<","<<par_sa*100<<"\n";
    for(int i=0;i<14;i++)
        f<<rCL[i]<<","<<rCA[i]<<","<<rSL[i]<<","<<rSA[i]<<"\n";
    f.close();

    return 0;
}
