#ifndef TESTHELPERS_HPP_INCLUDED
#define TESTHELPERS_HPP_INCLUDED

#include <streambuf>
#include <ostream>
#include <Eigen/Dense>

struct Err_p {
  int _count = 0;
  bool _inc = false;
  Err_p operator++(int) 
  {
    Err_p tmp(*this); 
    _count++; 
    _inc = true; 
    return tmp;
  }
};

std::ostream& operator<<(std::ostream& out, Err_p &err_p) {
  if (err_p._inc) {
    err_p._inc = false;
    return out << "\033[1;31m Unexpected\033[0m";
  }
  return out;
}

Eigen::MatrixXd prune(const Eigen::MatrixXd & A, double threshold = 1e-9) {
  Eigen::MatrixXd M = A;
  for (int i = 0; i < M.size();++i) {
    double &x = M.data()[i];
    if (std::abs(x) < threshold) {
      x = 0.;
    }
  }
  return M;
}

#define PRINTSIZE(X) X.rows()<<", "<<X.cols()

class NullStream : public std::ostream {
    class NullBuffer : public std::streambuf {
    public:
        int overflow( int c ) { return c; }
    } m_nb;
public:
    NullStream() : std::ostream( &m_nb ) {}
};

#endif 

