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
  bool _JZero = false;
  virtual ~Solution(){;};
};

// Notice: This solution is not smooth, B is continuous but not differentiable on two lines
// B is the Euclidean distance to (t,1/2), and is not differentiable along y at (y = 0), and along x at (x = t + 1/2).
struct Solution0 final : public Solution {
  Eigen::Vector<double,1> B (size_t map_id, const Eigen::Vector<double,2> &x) override 
  {
    if (_t > 1. || _t < 0.) _t -= std::floor(_t);
    const double X = x[0], Y = x[1];
    const double w = (X < _t - 0.5) ? 1 : ((X > _t + 0.5) ? -1 : 0);
    return Eigen::Vector<double,1>{2. + (Y-0.5)*(Y-0.5) + (X - _t + w)*(X - _t + w)};
  }
  Eigen::Vector<double,2> E (size_t map_id, const Eigen::Vector<double,2> &x) override 
  {
    if (_t > 1. || _t < 0.) _t -= std::floor(_t);
    const double X = x[0];
    const double w = (X < _t - 0.5) ? 1 : ((X > _t + 0.5) ? -1 : 0);
    return Eigen::Vector<double,2>{0., (X - _t + w)*(X - _t + w)};
  }
  Eigen::Vector<double,1> dE (size_t map_id, const Eigen::Vector<double,2> &x) override 
  {
    if (_t > 1. || _t < 0.) _t -= std::floor(_t);
    const double X = x[0];
    const double w = (X < _t - 0.5) ? 1 : ((X > _t + 0.5) ? -1 : 0);
    return Eigen::Vector<double,1>{2.*(X - _t + w)};
  }
  Eigen::Vector<double,1> rho (size_t map_id, const Eigen::Vector<double,2> &x) override 
  {
    if (_t > 1. || _t < 0.) _t -= std::floor(_t);
    return Eigen::Vector<double,1>{0.};
  }
  Eigen::Vector<double,2> J (size_t map_id, const Eigen::Vector<double,2> &x) override 
  {
    if (_t > 1. || _t < 0.) _t -= std::floor(_t);
    const double Y = x[1];
    return Eigen::Vector<double,2>{2*Y - 1.,0};
  }
};

#endif

