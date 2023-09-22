#ifndef SPHERE_REF_HPP
#define SPHERE_REF_HPP

#include <Eigen/Dense>

struct Solution {
  virtual Eigen::Vector<double,1> B (size_t map_id, const Eigen::Vector<double,2> &) = 0;
  virtual Eigen::Vector<double,2> E (size_t map_id, const Eigen::Vector<double,2> &) = 0;
  virtual Eigen::Vector<double,1> dE (size_t map_id, const Eigen::Vector<double,2> &) = 0;
  virtual Eigen::Vector<double,1> rho (size_t map_id, const Eigen::Vector<double,2> &) = 0;
  virtual Eigen::Vector<double,2> J (size_t map_id, const Eigen::Vector<double,2> &) = 0;
  double _t = 0.;
  virtual ~Solution(){;};
};

struct Solution0 final : public Solution {
  Eigen::Vector<double,1> B (size_t map_id, const Eigen::Vector<double,2> &x) override 
  {
    const double X = x[0], Y = x[1];
    const double ct = std::cos(_t), st = std::sin(_t);
    if (map_id == 0) {
      return Eigen::Vector<double,1>{(X*X + Y*Y - 1)*ct + X*X + Y*Y + 1 - 2*X*st};
    } else {
      return Eigen::Vector<double,1>{(X*X + Y*Y - 1)*ct - X*X - Y*Y - 1 + 2*X*st};
    }
  }
  Eigen::Vector<double,2> E (size_t map_id, const Eigen::Vector<double,2> &x) override 
  {
    const double X = x[0], Y = x[1];
    const double ct = std::cos(_t), st = std::sin(_t);
    if (map_id == 0) {
      return Eigen::Vector<double,2>{
        Y*(-2*X*ct + (-X*X - Y*Y + 2)*st)/4,
        X*(X*X + Y*Y - 2)*st/4 + (3*X*X + Y*Y - 3)*ct/4
      };
    } else {
      return Eigen::Vector<double,2>{
        Y*(2*X*ct + (-X*X - Y*Y + 2)*st)/4,
        X*(X*X + Y*Y - 2)*st/4 - (3*X*X + Y*Y - 3)*ct/4
      };
    }
  }
  Eigen::Vector<double,1> dE (size_t map_id, const Eigen::Vector<double,2> &x) override 
  {
    const double X = x[0], Y = x[1];
    const double ct = std::cos(_t), st = std::sin(_t);
    if (map_id == 0) {
      return Eigen::Vector<double,1>{(X*X + Y*Y - 1)*st + 2*X*ct};
    } else {
      return Eigen::Vector<double,1>{(X*X + Y*Y - 1)*st - 2*X*ct};
    }
  }
  Eigen::Vector<double,1> rho (size_t map_id, const Eigen::Vector<double,2> &x) override 
  {
    return Eigen::Vector<double,1>{0.};
  }
  Eigen::Vector<double,2> J (size_t map_id, const Eigen::Vector<double,2> &x) override 
  {
    const double X = x[0], Y = x[1];
    const double ct = std::cos(_t), st = std::sin(_t);
    if (map_id == 0) {
      double tmpJN = 3*(std::pow(X*X + Y*Y,2)/2. + X*X + Y*Y + 0.5) 
              + (1.5*std::pow(X*X + Y*Y,2) + 1.25*(X*X + Y*Y) - 1)*ct;
      return Eigen::Vector<double,2>{
          Y*tmpJN - X*Y*(2*X*X + 2*Y*Y + 2.5)*st,
         -X*tmpJN + (10*std::pow(X,4) + 12*X*X*Y*Y + 15*X*X + 2*std::pow(Y,4) + 5*Y*Y - 1)*st/4.
      };
    } else {
      double tmpJS = 3*(std::pow(X*X + Y*Y,2)/2. + X*X + Y*Y + 0.5) 
              - (1.5*std::pow(X*X + Y*Y,2) + 1.25*(X*X + Y*Y) - 1)*ct;
      return Eigen::Vector<double,2>{
         -Y*tmpJS + X*Y*(2*X*X + 2*Y*Y + 2.5)*st,
         X*tmpJS - (10*std::pow(X,4) + 12*X*X*Y*Y + 15*X*X + 2*std::pow(Y,4) + 5*Y*Y - 1)*st/4.
      };
    }
  }
};

struct Solution1 final : public Solution {
  Eigen::Vector<double,1> B (size_t map_id, const Eigen::Vector<double,2> &x) override 
  {
    const double X = x[0], Y = x[1];
    const double ct = std::cos(_t), st = std::sin(_t);
    if (map_id == 0) {
      return Eigen::Vector<double,1>{(X*X + Y*Y - 1)*ct + X*X + Y*Y + 1 - 2*X*st}
              *std::pow(2./(1.+X*X+Y*Y),3);
    } else {
      return Eigen::Vector<double,1>{(X*X + Y*Y - 1)*ct - X*X - Y*Y - 1 + 2*X*st}
              *std::pow(2./(1.+X*X+Y*Y),3);
    }
  }
  Eigen::Vector<double,2> E (size_t map_id, const Eigen::Vector<double,2> &x) override 
  {
    return Eigen::Vector<double,2>{0.,0.};
  }
  Eigen::Vector<double,1> dE (size_t map_id, const Eigen::Vector<double,2> &x) override 
  {
    return Eigen::Vector<double,1>{0.};
  }
  Eigen::Vector<double,1> rho (size_t map_id, const Eigen::Vector<double,2> &x) override 
  {
    return Eigen::Vector<double,1>{0.};
  }
  Eigen::Vector<double,2> J (size_t map_id, const Eigen::Vector<double,2> &x) override 
  {
    return Eigen::Vector<double,2>{0.,0.};
  }
};

struct Solution2 final : public Solution {
  Eigen::Vector<double,1> B (size_t map_id, const Eigen::Vector<double,2> &x) override 
  {
    const double X = x[0], Y = x[1];
    const double ct = std::cos(_t), st = std::sin(_t);
    if (map_id == 0) {
      return Eigen::Vector<double,1>{(X*X + Y*Y - 1)*ct + X*X + Y*Y + 1 - 2*X*st}
              *std::pow(2./(1.+X*X+Y*Y),3);
    } else {
      return Eigen::Vector<double,1>{(X*X + Y*Y - 1)*ct - X*X - Y*Y - 1 + 2*X*st}
              *std::pow(2./(1.+X*X+Y*Y),3);
    }
  }
  Eigen::Vector<double,2> E (size_t map_id, const Eigen::Vector<double,2> &x) override 
  {
    if (map_id == 0) {
      return Eigen::Vector<double,2>{1.,0.};
    } else {
      return Eigen::Vector<double,2>{1.,0.};
    }
  }
  Eigen::Vector<double,1> dE (size_t map_id, const Eigen::Vector<double,2> &x) override 
  {
    return Eigen::Vector<double,1>{0.};
  }
  Eigen::Vector<double,1> rho (size_t map_id, const Eigen::Vector<double,2> &x) override 
  {
    return Eigen::Vector<double,1>{0.};
  }
  Eigen::Vector<double,2> J (size_t map_id, const Eigen::Vector<double,2> &x) override 
  {
    return Eigen::Vector<double,2>{0.,0.};
  }
};

struct Solution3 final : public Solution {
  Eigen::Vector<double,1> B (size_t map_id, const Eigen::Vector<double,2> &x) override 
  {
    return Eigen::Vector<double,1>{0.};
    //return ((map_id == 1)? -1 : 1)*std::pow(2./(1+x[0]*x[0]+x[1]*x[1]),2)*rho(map_id,x);
  }
  Eigen::Vector<double,2> E (size_t map_id, const Eigen::Vector<double,2> &x) override 
  {
    return Eigen::Vector<double,2>{0.,0.};
  }
  Eigen::Vector<double,1> dE (size_t map_id, const Eigen::Vector<double,2> &x) override 
  {
    return Eigen::Vector<double,1>{0.};
  }
  Eigen::Vector<double,1> rho (size_t map_id, const Eigen::Vector<double,2> &x) override 
  {
    const double X = x[0], Y = x[1];
    constexpr double t_0 = 0.8;
    double ct = std::cos(t_0), st = std::sin(t_0);
    double r2 = X*X + Y*Y;
    if (map_id == 0) {
      double val = 2.*(1. - (2.*X*st + (1. - r2)*ct)/(1. + r2));
      return Eigen::Vector<double,1>{(val > 3.8)? val : 0.};
    } else {
      double val = 2.*(1. - (2.*X*st - (1. - r2)*ct)/(1. + r2));
      return Eigen::Vector<double,1>{(val > 3.8)? val : 0.};
    }
  }
  Eigen::Vector<double,2> J (size_t map_id, const Eigen::Vector<double,2> &x) override 
  {
    return Eigen::Vector<double,2>{0.,0.};
  }
};

#endif

