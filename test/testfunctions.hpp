#ifndef TESTFUNCTIONS_HPP_INCLUDED
#define TESTFUNCTIONS_HPP_INCLUDED

#include <Eigen/Dense>
#include "exterior_dimension.hpp"

class Poly_One_base {
  public:
    Poly_One_base(int r) : _r(r){;}

    // Initial
    double scalar(Eigen::Vector2d const & x) const {
      double rv = 0.;
      for (int h = 0; h <= _r; ++h) {// harmonic degree
        for (int i = 0; i <= h; ++i) {
          rv += std::pow(x(0),i)*std::pow(x(1),h-i);
        }
      }
      return rv;
    }
    // Differentials
    double dx(Eigen::Vector2d const & x) const {
      double rv = 0.;
      for (int h = 1; h <= _r; ++h) {
        for (int i = 1; i <= h; ++i) {
          rv += i*std::pow(x(0),i-1)*std::pow(x(1),h-i);
        }
      }
      return rv;
    }
    double dy(Eigen::Vector2d const & x) const {
      double rv = 0.;
      for (int h = 1; h <= _r; ++h) {
        for (int i = 0; i < h; ++i) {
          rv += (h-i)*std::pow(x(0),i)*std::pow(x(1),h-i-1);
        }
      }
      return rv;
    }

  private:
    int _r;
};

class Poly_One : public Poly_One_base {
  public:
    Poly_One(int r) : Poly_One_base(r) {;}

    double P0(Eigen::Vector2d const & x) const {
      return scalar(x);
    }
    Eigen::Vector2d P1(Eigen::Vector2d const & x) const {
      double rv = scalar(x);
      return {rv,2.*rv};
    }
    double P2(Eigen::Vector2d const & x) const {
      return scalar(x);
    }
    // Differentials
    Eigen::Vector2d d0(Eigen::Vector2d const & x) const {
      return {dx(x),dy(x)};
    }
    double d1(Eigen::Vector2d const &x) const {
      return 2*dx(x) - dy(x);
    }
};

class Poly_One_trimmed : public Poly_One_base {
  public:
    Poly_One_trimmed(int r) : Poly_One_base(r-1) {;}

    double P0(Eigen::Vector2d const & x) const {
      return x(0)*scalar(x);
    }
    Eigen::Vector2d P1(Eigen::Vector2d const &x) const {
      double alpha = scalar(x);
      Eigen::Vector2d rv{- x(1),x(0) + 2.};
      return alpha*rv;
    }
    double P2(Eigen::Vector2d const & x) const {
      return scalar(x);
    }
    // Differentials
    Eigen::Vector2d d0(Eigen::Vector2d const & x) const {
      return {scalar(x) + x(0)*dx(x),x(0)*dy(x)};
    }
    double d1(Eigen::Vector2d const &x) const {
      double px = dx(x), py = dy(x), a = scalar(x);
      return 2.*a + x(1)*py + x(0)*px + 2.*px;
    }
};

template<typename P,typename T>
class PolyPullback {
  public:
    PolyPullback(P const & Poly,T const & Tcell) : _Poly(Poly), _Tcell(Tcell) {;}
    template<size_t k>
    Eigen::Vector<double,Manicore::Dimension::ExtDim(k,2)> P_ev(Eigen::Vector2d const & x) const {
      if constexpr (k == 0) {
        return _Tcell.template evaluate_DJ_p<0>(0,x)*_Poly.P0(_Tcell.evaluate_J(0,x));
      } else if constexpr (k == 1) {
        return _Tcell.template evaluate_DJ_p<1>(0,x)*_Poly.P1(_Tcell.evaluate_J(0,x));
      } else if constexpr (k == 2) {
        return _Tcell.template evaluate_DJ_p<2>(0,x)*_Poly.P2(_Tcell.evaluate_J(0,x));
      }
    }
    // Differentials
    template<size_t k>
    Eigen::Vector<double,Manicore::Dimension::ExtDim(k+1,2)> D_ev(Eigen::Vector2d const & x) const {
      if constexpr (k == 0) {
        return _Tcell.template evaluate_DJ_p<1>(0,x)*_Poly.d0(_Tcell.evaluate_J(0,x));
      } else if constexpr (k == 1) {
        return _Tcell.template evaluate_DJ_p<2>(0,x)*_Poly.d1(_Tcell.evaluate_J(0,x));
      } else if constexpr (k == 2) {
        return Eigen::Vector<double,0>{};
      }
    }
  private:
    P const & _Poly;
    T const & _Tcell;
};

#endif

