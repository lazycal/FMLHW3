#include <cstdio>
#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>
#include <limits>
#include <fstream>
#include <cassert>
#include <tuple>
#include <numeric>
#include <array>

using std::cout;
using std::endl;
const int M=5000, N=10;
const double EPS=1e-6;
inline int sgn(double x){return x>=0?1:-1;}
template <class T>
std::ostream& operator<<(std::ostream &os, const std::vector<T>&a)
{
  os << "[";
  for (const auto &i: a) os << i << ",";
  os << "]";
  return os;
}

struct Base {
  double thres;
  int sgn, col; // predict sgn if x >= thres
  // int predict(double x);
  int predict(const double *x) const {
    // cout << "predicting " << x[col] << " thres=" << thres << "prediction=" << ::sgn(x[col]-thres) * this->sgn << "\n";
    return ::sgn(x[col]-thres) * this->sgn;
  }
  friend std::ostream& operator<<(std::ostream &os, const Base &o);
  friend std::istream& operator>>(std::istream &is, Base &o);
};
std::istream& operator>>(std::istream &is, Base &o)
{
  is >> o.thres >> o.sgn >> o.col;
  return is;
}
std::ostream& operator<<(std::ostream &os, const Base &o)
{
  os << o.thres << " " << o.sgn << " " << o.col << " ";
  return os;
}

struct Ensemble {
  int n;
  std::vector<std::pair<Base, double> > data;
  void emplace_back(Base a, double b) {data.emplace_back(a, b);}
  int predict(const double *x) const { // TODO: optimize
    double score = 0;
    for (const auto &h: data) {
      score += h.first.predict(x) * h.second;
    }
    return sgn(score);
  }
  Ensemble(int n=-1):n(n) {}
  void dump(std::string file) {
    std::ofstream f(file);
    f << *this;
  }
  void load(std::string file) {
    std::ifstream f(file);
    f >> *this;
  }
  friend std::ostream& operator<<(std::ostream &os, const Ensemble &o);
  friend std::istream& operator>>(std::istream &is, Ensemble &o);
};
std::istream& operator>>(std::istream &is, Ensemble &o) {
  o.data.clear();
  int T;
  is >> o.n >> T;
  for (int i = 0; i < T; ++i) {
    Base h; double a;
    is >> h >> a;
    o.emplace_back(h, a);
  }
  return is;
}
std::ostream& operator<<(std::ostream &os, const Ensemble &o)
{
  os << o.n << " " << o.data.size() << "\n";
  for (const auto &h: o.data) os << h.first << " " << h.second << "\n";
  return os;
}

double eval(const Ensemble &h, const double (&X)[M][N], const int (&y)[M],
  int m, int n) // return error rate
{
  double res = 0;
  for (int i = 0; i < m; ++i) {
    res += (h.predict(X[i]) != y[i]);
  }
  return res / m;
}
double eval(const Base &h, const double (&X)[M][N], const int (&y)[M],
  int m, int n, const std::vector<double> &D)
{
  double res = 0;
  for (int i = 0; i < m; ++i) {
    // cout << "y[i]="<<y[i] << ", ";
    res += (h.predict(X[i]) != y[i]) * D[i];
  }
  return res;
}

std::pair<Base, double> pick_min(const double (&X)[M][N], const int (&y)[M],
  int m, int n, const std::vector<double> &y_sorted, const std::vector<int> &idx, 
  const std::vector<double> &D) 
{
  // {
  // auto h = Base{std::numeric_limits<double>::infinity(), -1, 0};
  // double e = eval(h, X, y, m, n, D);
  // return {h, e};
  // }
  Base min_h;
  double min_e=std::numeric_limits<double>::infinity();
  for (int j = 0; j < n; ++j) {
    std::vector<double> fea(m);
    for (int i = 0; i < m; ++i) fea[i] = X[i][j];
    std::sort(fea.begin(), fea.end());
    int p = std::unique(fea.begin(), fea.end()) - fea.begin();
    // cout << "p=" << p << endl;
    for (int i = 0; i < p; ++i) {
      for (int s = -1; s <= 1; s += 2) {
        Base h{i == 0 ? std::numeric_limits<double>::lowest() : fea[i], s, j};
        double err = eval(h, X, y, m, n, D);
        if (min_e > err) {
          min_e = err;
          min_h = h;
        }
      }
    }
  }
  return {min_h, min_e};
}

template<class It>
void normalize(It be, It ed)
{
  auto s = *be * 0;
  for (auto it = be; it < ed; ++it) {
    s += *it;
    assert(*it>=0);
  }
  for (auto it = be; it < ed; ++it) *it = *it / s;
}

Ensemble AdaBoostLog(double (&X)[M][N], int (&y)[M],
  int m, int n, int T, bool use_log, 
  double (&X_test)[M][N], int (&y_test)[M], int m_test)
{
  std::vector<double> D(m, 1./m), fx(m);
  Ensemble res(n);
  // pre sorting
  std::vector<double> y_sorted(n*m);
  std::vector<int> idx(n*m);
  for (int j = 0; j < n; ++j) {
    auto idx_j = idx.begin() + j*m;
    for (int i = 0; i < m; ++i) idx_j[i] = i;
    std::sort(idx_j, idx_j + m, [&](int lhs, int rhs) {
      return X[lhs][j] < X[rhs][j];
    });
    for (int i = 0; i < m; ++i) y_sorted[j*m + i] = y[idx_j[i]];
  }

  for (int t = 1; t <= T; ++t) {
    Base h; double e, alpha;
    std::tie(h, e) = pick_min(X, y, m, n, y_sorted, idx, D);
    alpha = 0.5*std::log((1-e)/e);
    // update f_t(x_i)
    for (int i = 0; i < m; ++i) {
      fx[i] += h.predict(X[i]) * alpha;
      D[i] = use_log ? (1. / (1+std::exp(y[i] * fx[i]))) : 
        (std::exp(-alpha * y[i] * h.predict(X[i])) * D[i]);
    }
    // normalize D
    normalize(D.begin(), D.end());
    assert(std::abs(std::accumulate(D.begin(), D.end(), 0.) - 1) < EPS);
    res.emplace_back(h, alpha);
    if (t % 1 == 0) {
      double test_err = eval(res, X_test, y_test, m_test, n);
      cout << "t=#" << t << ": error=" << e << ", test error=" << test_err << endl;
    }
  }
  return res;
}

double X_train[M][N], X_test[M][N];
int y_train[M], y_test[M];

void read_data(std::string file, double (&X)[M][N], int (&y)[M], int &m, int &n)
{
  std::ifstream f(file);
  n = 0; m = -1;
  while (1) {
    std::string s;
    if (!(f >> s)) break;
    auto pos = std::find(s.begin(), s.end(), ':');
    if (pos != s.end()) {
      X[m][n++] = std::stod(std::string(pos+1, s.end()));
    } else {
      y[++m] = std::stoi(s);
      assert(y[m] == 0 || y[m] == 1);
      y[m] = 2 * y[m] - 1;
      n = 0;
    }
  }
  ++m;
  // for (int i = 0; i < m; ++i) X[i][0] = y[i];
  // cout << "y,X=\n";
  // for (int i = 0; i < m; ++i) {
  //   cout << y[i] << " ";
  //   for (int j = 0; j < n; ++j) cout << j+1 << ":" << X[i][j] << "\n "[j<n-1];
  // }
}
int main(int argc, char* argv[])
{
  // usage: main /path/to/train /path/to/test T /path/to/save/model use_log
  int m_train, m_test, n;
  assert(argc == 6);
  read_data(argv[1], X_train, y_train, m_train, n);
  read_data(argv[2], X_test, y_test, m_test, n);
  assert(n == 10);
  int T = std::stoi(argv[3]);
  bool use_log = std::stoi(argv[5]);
  cout << "m_train=" << m_train << ", m_test=" << m_test << ", T=" << T << 
    ", use_log=" << use_log << "\n";
  assert(T%100==0);
  auto ens = AdaBoostLog(X_train, y_train, m_train, n, T, use_log, X_test, y_test, m_test);
  double error = eval(ens, X_test, y_test, m_test, n);
  cout << "error=\n" << error << "\n";
  std::string save_path = argv[4];
  ens.dump(save_path);

  // test
  cout << "testing..." << endl;
  ens.load(save_path);
  ens.dump(save_path+"unittest");
  assert(system(("diff "+save_path+" "+save_path+"unittest").c_str()) == 0);
  Ensemble ens1;
  ens1.load(save_path);
  assert(eval(ens1, X_test, y_test, m_test, n) == error);
}