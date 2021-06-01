#include "json/json.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <map>

#include <unistd.h>

int testST, testED;
const int W = 8;
const int H = 8;
const int S = W * H;
const int C = 1024;
const int MAXN = 100;
const int MAXSIZE = 100;
const int VARNUM = 20000;
const int CONNUM = 2000000;
const int ENERGY_COMPUTE = 1;
const int ENERGY_TRANSFER = 1;
const double sigma = 0.5;
int x[S];
int y[S];
std::map<Json::Value, int> coreNum;
std::map<Json::Value, Json::Value> axonToCore;
std::map<Json::Value, Json::Value> neuronToCore;
Json::Value numCore[MAXSIZE];
Json::Int axonSize[MAXSIZE];
Json::Int neuronSize[MAXSIZE];
int l[MAXSIZE];
int data[MAXSIZE][MAXSIZE];
int N;
std::vector<std::pair<int, int> > eq[CONNUM];
std::vector<std::pair<int, int> > leq[CONNUM];
int eqCon[CONNUM];
int leqCon[CONNUM];
int eqNum, leqNum;
int varNum, binNum;
double a[VARNUM], b[VARNUM], c[VARNUM], d[VARNUM], e[VARNUM], f[VARNUM];
double tg[VARNUM], ptg[VARNUM];
int rat[VARNUM];

void init() {
    int cnt = 0;
    for (int i = 0; i < W; i++)
        for (int j = 0; j < H; j++) {
            x[cnt] = i;
            y[cnt++] = j;
        }

    std::ifstream ifs("info.txt");
    Json::Reader reader;
    Json::Value root;

    if (!reader.parse(ifs, root, false)) {
        ifs.close();
        return;
    }
    ifs.close();

    N = root["cores"].size();
    for (int i = 0; i < N; i++) {
        Json::Value core = root["cores"][i];
        Json::Value coreName = core["core_name"];
        coreNum[coreName] = i;
        numCore[i] = coreName;
        axonSize[i] = core["axon_num"].asInt();
        l[i] = neuronSize[i] = core["neuron_num"].asInt();
        for (int j = 0; j < axonSize[i]; j++)
            axonToCore[core["axons"][j]["axon_name"]] = coreName;
    }

    memset(data, 0, sizeof data);
    for (int i = 0; i < N; i++) {
        Json::Value core = root["cores"][i];
        int neuronCnt = core["neurons"].size();
        for (int j = 0 ; j < neuronCnt; j++) {
            Json::Value neuron = core["neurons"][j];
            Json::Value coreName = core["core_name"];
            neuronToCore[neuron["neuron_name"]] = coreName;
            int xi = coreNum[core["core_name"]];
            if (neuron["route_to"] == "C-1A-1") continue;
            int yi = coreNum[axonToCore[neuron["route_to"]]];
            data[xi][yi]++;
        }
    }

    std::cout << N << std::endl;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (data[i][j] > 0 && data[j][i] > 0)
                std::cout << i << ' ' << j << std::endl;
}

static int map[MAXN][S]; // if neuron_i mapped into core_j
static int aux[MAXN][MAXN]; // auxiliary variables
static int fin[MAXN];    // finish time of neuron_i
static int lat[MAXN][MAXN]; // data transfer latency from neuron_i to neuron_j
static int clat[MAXN];   // computation latency of neuron_i

void createModel() {
    std::cout << "N = " << N << std::endl;
    varNum = 0;
    memset(rat, 0, sizeof rat);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < S; j++)
            map[i][j] = varNum++;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            aux[i][j] = varNum++;
    binNum = varNum;
    for (int i = 0; i < N; i++)
        fin[i] = varNum++;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            if (data[i][j] > 0) rat[varNum] = ENERGY_TRANSFER;
            lat[i][j] = varNum++;
        }
    for (int i = 0; i < N; i++) {
        //rat[varNum] = ENERGY_COMPUTE;
        clat[i] = varNum++;
    }

    std::cout << "varNum = " << varNum << std::endl;

    eqNum = leqNum = 0;

    // no copy constraint
    for (int i = 0; i < N; i++) {
        eqCon[++eqNum] = 1;
        eq[eqNum].clear();
        for (int j = 0; j < S; j++) {
            eq[eqNum].push_back(std::make_pair(map[i][j], 1));
            leqCon[++leqNum] = 0;
            leq[leqNum].clear();
            leq[leqNum].push_back(std::make_pair(map[i][j], -1));
            leqCon[++leqNum] = 1;
            leq[leqNum].clear();
            leq[leqNum].push_back(std::make_pair(map[i][j], 1));
        }
    }

    std::cout << "constraint 1 done, leqNum = " << leqNum << std::endl;

    // core capacity constraint
    for (int j = 0; j < S; j++) {
        leqCon[++leqNum] = C;
        leq[leqNum].clear();
        for (int i = 0; i < N; i++) leq[leqNum].push_back(std::make_pair(map[i][j], neuronSize[i]));
        leqCon[++leqNum] = C;
        leq[leqNum].clear();
        for (int i = 0; i < N; i++) leq[leqNum].push_back(std::make_pair(map[i][j], axonSize[i]));
    }

    std::cout << "constraint 2 done, leqNum = " << leqNum << std::endl;

    // latency constraint
    for (int i = 0; i < N; i++) {
        leqCon[++leqNum] = 0;//-neuronSize[i];
        leq[leqNum].clear();
        leq[leqNum].push_back(std::make_pair(clat[i], -1));

        leqCon[++leqNum] = 0;
        leq[leqNum].clear();
        leq[leqNum].push_back(std::make_pair(clat[i], 1));
        leq[leqNum].push_back(std::make_pair(fin[i], -1));
    }

    testST = leqNum;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (data[i][j] > 0) {
                leqCon[++leqNum] = 0;
                leq[leqNum].clear();
                leq[leqNum].push_back(std::make_pair(lat[i][j], -1));
                for (int k = 0; k < S; k++) {
                    leq[leqNum].push_back(std::make_pair(map[i][k], (x[k] + y[k]) * data[i][j] * 2));
                    leq[leqNum].push_back(std::make_pair(map[j][k], -(x[k] + y[k]) * data[i][j] * 2));
                }
                leqCon[++leqNum] = 0;
                leq[leqNum].clear();
                leq[leqNum].push_back(std::make_pair(lat[i][j], -1));
                for (int k = 0; k < S; k++) {
                    leq[leqNum].push_back(std::make_pair(map[i][k], (-x[k] + y[k]) * data[i][j] * 2));
                    leq[leqNum].push_back(std::make_pair(map[j][k], (x[k] - y[k]) * data[i][j] * 2));
                }
                leqCon[++leqNum] = 0;
                leq[leqNum].clear();
                leq[leqNum].push_back(std::make_pair(lat[i][j], -1));
                for (int k = 0; k < S; k++) {
                    leq[leqNum].push_back(std::make_pair(map[i][k], (x[k] - y[k]) * data[i][j] * 2));
                    leq[leqNum].push_back(std::make_pair(map[j][k], (-x[k] + y[k]) * data[i][j] * 2));
                }
                leqCon[++leqNum] = 0;
                leq[leqNum].clear();
                leq[leqNum].push_back(std::make_pair(lat[i][j], -1));
                for (int k = 0; k < S; k++) {
                    leq[leqNum].push_back(std::make_pair(map[i][k], -(x[k] + y[k]) * data[i][j] * 2));
                    leq[leqNum].push_back(std::make_pair(map[j][k], (x[k] + y[k]) * data[i][j] * 2));
                }
                leqCon[++leqNum] = data[i][j] * 4;
                leq[leqNum].clear();
                leq[leqNum].push_back(std::make_pair(lat[i][j], 1));
            }
        
    testED = leqNum;

    std::cout << "constraint 3 done, leqNum = " << leqNum << std::endl;

    // serial physical core constraint
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            if (i != j)
                for (int k = 0; k < S; k++) {
                    leqCon[++leqNum] = 2000000;
                    leq[leqNum].clear();
                    leq[leqNum].push_back(std::make_pair(aux[i][j], -100000));
                    leq[leqNum].push_back(std::make_pair(map[i][k], 1000000));
                    leq[leqNum].push_back(std::make_pair(map[j][k], 1000000));
                    leq[leqNum].push_back(std::make_pair(fin[i], -1));
                    leq[leqNum].push_back(std::make_pair(fin[j], 1));
                    leq[leqNum].push_back(std::make_pair(clat[i], 1));

                    leqCon[++leqNum] = 2100000;
                    leq[leqNum].clear();
                    leq[leqNum].push_back(std::make_pair(aux[i][j], 100000));
                    leq[leqNum].push_back(std::make_pair(map[i][k], 1000000));
                    leq[leqNum].push_back(std::make_pair(map[j][k], 1000000));
                    leq[leqNum].push_back(std::make_pair(fin[i], 1));
                    leq[leqNum].push_back(std::make_pair(fin[j], -1));
                    leq[leqNum].push_back(std::make_pair(clat[j], 1));

                    leqCon[++leqNum] = 0;
                    leq[leqNum].clear();
                    leq[leqNum].push_back(std::make_pair(aux[i][j], -1));

                    leqCon[++leqNum] = 1;
                    leq[leqNum].clear();
                    leq[leqNum].push_back(std::make_pair(aux[i][j], 1));
                }
    }

    std::cout << "constraint 4 done, leqNum = " << leqNum << std::endl;

    // virtual core order constraint
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (data[i][j] > 0) {
                leqCon[++leqNum] = 0;
                leq[leqNum].clear();
                leq[leqNum].push_back(std::make_pair(fin[i], 1));
                leq[leqNum].push_back(std::make_pair(lat[i][j], 1));
                leq[leqNum].push_back(std::make_pair(clat[j], 1));
                leq[leqNum].push_back(std::make_pair(fin[j], -1));
            }

    std::cout << "constraint 5 done, leqNum = " << leqNum << std::endl;
}

double tau = 0.5;
double eps = 0.05;
double alp, bet, gam;

void Pi(double (&x)[VARNUM], int i) {
    double calc = 0;
    long long len = 0;
    for (std::vector<std::pair<int, int> >::iterator po = eq[i].begin(); po != eq[i].end(); po++) {
        calc += (double)(*po).second * x[(*po).first];
        len += (long long)(*po).second * (*po).second;
    }
    calc = (eqCon[i] - calc) / len;
    for (std::vector<std::pair<int, int> >::iterator po = eq[i].begin(); po != eq[i].end(); po++)
        x[(*po).first] += sigma * calc * (*po).second;
}

void Pj(double (&x)[VARNUM], int i) {
    double calc = 0;
    long long len = 0;
    for (std::vector<std::pair<int, int> >::iterator po = leq[i].begin(); po != leq[i].end(); po++) {
        calc += (double)(*po).second * x[(*po).first];
        len += (long long)(*po).second * (*po).second;
    }
    calc = (leqCon[i] - calc) / len;
    calc = std::fmin((double)0, calc);
    for (std::vector<std::pair<int, int> >::iterator po = leq[i].begin(); po != leq[i].end(); po++)
        x[(*po).first] += sigma * calc * (*po).second;
}

void P(double (&x)[VARNUM], double (&y)[VARNUM]) {
    for (int i = 0; i < varNum; i++) x[i] = y[i];
    for (int i = 0; i < eqNum; i++) Pi(x, i);
    for (int i = 0; i < leqNum; i++) Pj(x, i);
}

double tot(double (&x)[VARNUM]) {
    double ret = 0;
    for (int i = 0; i < varNum; i++) ret += rat[i] * x[i];
//    for (int i = 0; i < N; i++) ret = std::fmax(ret, x[fin[i]]);
    return ret;
}

void t(double (&x)[VARNUM], double (&y)[VARNUM]) {
    for (int i = 0; i < varNum; i++) x[i] = y[i] - gam * rat[i];
    // for (int i = 0; i < binNum; i++) x[i] = y[i] - gam * rat[i];
    // for (int i = binNum; i < varNum; i++) x[i] = y[i] - 100 * gam * rat[i];

    // for (int i = 0; i < varNum; i++) x[i] = y[i];
    // for (int i = 0; i < N; i++) x[fin[i]] = y[fin[i]] - 100 * gam;
}

void T(double (&x)[VARNUM], double (&y)[VARNUM], double lambda) {
    t(tg, y);
    P(ptg, tg);
    for (int i = 0; i < varNum; i++) x[i] = ptg[i] * lambda + tg[i] * (1 - lambda);
}

double r(double (&x)[VARNUM]) {
    double ret = 0;
    double sum;
    for (int i = 0; i < eqNum; i++) {
        sum = 0;
        for (std::vector<std::pair<int, int> >::iterator po = eq[i].begin(); po != eq[i].end(); po++)
            sum += x[(*po).first] * (*po).second;
        sum -= eqCon[i];
        ret += sum * sum;
    }
    for (int i = 0; i < leqNum; i++) {
        sum = 0;
        for (std::vector<std::pair<int, int> >::iterator po = leq[i].begin(); po != leq[i].end(); po++)
            sum += x[(*po).first] * (*po).second;
        sum -= leqCon[i];
        sum = std::fmax((double)0, sum);
        ret += sum * sum;
    }
    return std::sqrt(ret);
}

void calc(int lim, double lambda) {
    memset(a, 0, sizeof a);
    for (int i = binNum; i < varNum; i++)
        a[i] = 1000;
    
    bool loop;
    for (int i = 1; i <= lim; i++) {
        alp = tau * i / lim;
        bet = 1 - (1 - tau) * i / lim;

        // binaryzation
        // for (int j = 0; j < binNum; j++)
        //     if (a[j] <= alp) e[j] = b[j] = 0;
        //     else if (a[j] >= bet) e[j] = b[j] = 1;
        //     else e[j] = b[j] = a[j];
        // for (int j = binNum; j < varNum; j++)
        //     e[j] = b[j] = a[j];
       for (int j = 0; j < varNum; j++) e[j] = b[j] = a[j];

//        std::cout << "binaryzation done" << std::endl;

        // superiority
        gam = 1;
        loop = true;
        while (loop && gam > 1e-6) {
            t(c, b);
            if (tot(c) <= tot(b)) {
                T(d, c, lambda);
//                std::cout << std::fixed << std::setprecision(5) << r(b) << "->" << r(c) << "->" << r(d) << std::endl;
                if (r(d) < r(b)) loop = false;
                else gam = gam / 2;
            } else gam = gam / 2;
//            std::cout << std::fixed << std::setprecision(5) << "gam = " << gam << std::endl;
        }
        if (gam < 1e6 + 1e7) gam = 0;
        T(f, e, lambda);


        // conflict resolving
        // for (int j = 0; j < binNum; j++)
        //     if (a[j] <= alp && f[j] >= tau) c[j] = tau - eps;
        //     else if (a[j] >= bet && f[j] <= tau) c[j] = tau + eps;
        //     else c[j] = f[j];
        // for (int j = binNum; j < varNum; j++)
        //     c[j] = f[j];
            
       for (int j = 0; j < varNum; j++) c[j] = f[j];
    //    P(c, a);

        std::cout << std::fixed << std::setprecision(5) << i << "\'th conflict resolving done, r(x) = " << r(c) << ", totans = " << tot(c) << std::endl;

        for (int j = 0; j < varNum; j++) a[j] = c[j];
    }
}

void fin_calc() {
    for (int i = 0; i < binNum; i++)
        if (a[i] <= 0.5) a[i] = 0;
        else a[i] = 1;
}

void outp() {
    freopen("output.txt", "w", stdout);
    std::cout << "map" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < S; j++)
            std::cout << a[map[i][j]] << " ";
        std::cout << std::endl;
    }
/*    std::cout << "aux" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            std::cout << a[aux[i][j]] << " ";
        std::cout << std::endl;
    }*/
    std::cout << "fin" << std::endl;
    for (int i = 0; i < N; i++)
        std::cout << std::fixed << std::setprecision(3) << a[fin[i]] << " ";
    std::cout << std::endl;
    std::cout << "lat" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            std::cout << std::fixed << std::setprecision(3) << a[lat[i][j]] << " ";
        std::cout << std::endl;
    }
    std::cout << "clat" << std::endl;
    for (int i = 0; i < N; i++)
        std::cout << std::fixed << std::setprecision(3) << a[clat[i]] << " ";
    std::cout << std::endl << "a?" << std::endl;
    for (int i = 0; i < varNum; i++) std::cout << (a[i] <= 0.5? 0 : 1) << " ";
    std::cout << std::endl;
    std::cout << "r = " << r(a) << ", totans = " << tot(a) << std::endl;
    fclose(stdout);
}

int main() {
    std::ios::sync_with_stdio(false);
    init();
    std::cout << "init done" << std::endl;
    createModel();
    std::cout << "creat model done" << std::endl;
    calc(100, 1);
//    fin_calc();
    std::cout << "calc done" << std::endl;
    outp();
    std::cout << "outp done" << std::endl;
}