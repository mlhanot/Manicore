#include "map_interface.hpp"


struct circle_edge_map : public Manicore::ParametrizedMap<2,1> 
{
  Eigen::Vector<double,2> I(Eigen::Vector<double,1> const & t) const override final {
    double ratio = _extra[0];
    double R = _extra[1];
    double ca = _extra[2];
    double sa = _extra[3];
    double tr = t(0)*ratio*2.;
    double tmp1 = R/std::sqrt(1. + tr*tr);
    Eigen::Vector<double,2> ref_val{tr*tmp1,tmp1};
    return Eigen::Matrix<double,2,2>{{ca,-sa},{sa,ca}}*ref_val;
  }
  Eigen::Vector<double,1> J(Eigen::Vector<double,2> const & x) const override final {
    double ratio = _extra[0];
    double ca = _extra[2];
    double sa = _extra[3];
    Eigen::Vector<double,2> x_rot = Eigen::Matrix<double,2,2>{{ca,sa},{-sa,ca}}*x;
    double X = x_rot(0);
    double Y = x_rot(1);
    return Eigen::Vector<double,1>{X/(2.*Y*ratio)};
  }
};
struct circle_edge_pullbacks : public Manicore::ParametrizedDerivedMap<2,1> 
{
  Eigen::Matrix<double,2,1> DI(Eigen::Vector<double,1> const & x_in) const override final {
    double ratio = _extra[0];
    double R = _extra[1];
    double ca = _extra[2];
    double sa = _extra[3];
    double t = x_in(0);
    double tmp1 = R*2.*ratio/std::pow(4.*t*t*ratio*ratio+1.,1.5);
    Eigen::Vector<double,2> ref_val{tmp1,-2.*t*ratio*tmp1};
    return Eigen::Matrix<double,2,2>{{ca,-sa},{sa,ca}}*ref_val;
  }
  Eigen::Matrix<double,1,2> DJ(Eigen::Vector<double,2> const & x_in) const override final {
    double ratio = _extra[0];
    double ca = _extra[2];
    double sa = _extra[3];
    Eigen::Vector<double,2> x_rot = Eigen::Matrix<double,2,2>{{ca,sa},{-sa,ca}}*x_in;
    double X = x_rot(0);
    double Y = x_rot(1);
    double tmp1 = 1./(2.*Y*ratio);
    return Eigen::Matrix<double,1,2>{{tmp1,-X*tmp1/Y}}
           *Eigen::Matrix<double,2,2>{{ca,sa},{-sa,ca}};
  }
};

struct circle_face_map : public Manicore::ParametrizedMap<2,2> 
{
  Eigen::Vector<double,2> I(Eigen::Vector<double,2> const & x_in) const override final {
    double xT = _extra[0];
    double xB = _extra[1];
    double yT = _extra[2];
    double yB = _extra[3];
    double ca = _extra[4];
    double sa = _extra[5];
    double t = x_in(0);
    double p = x_in(1);
    double r = std::sqrt(yB*yB + t*t*4.*xB*xB);
    double R = std::sqrt(xT*xT+yT*yT);
    double tr = t*2.*xB/yB;
    double tmp1 = (p*R + (1. - p)*r)/std::sqrt(1.+tr*tr);
    Eigen::Vector<double,2> ref_val{tr*tmp1,tmp1};
    return Eigen::Matrix<double,2,2>{{ca,-sa},{sa,ca}}*ref_val;
  }
  Eigen::Vector<double,2> J(Eigen::Vector<double,2> const & x_in) const override final {
    double xT = _extra[0];
    double xB = _extra[1];
    double yT = _extra[2];
    double yB = _extra[3];
    double ca = _extra[4];
    double sa = _extra[5];
    Eigen::Vector<double,2> x_rot = Eigen::Matrix<double,2,2>{{ca,sa},{-sa,ca}}*x_in;
    double x = x_rot(0);
    double y = x_rot(1);
    double r = std::sqrt(x*x+y*y);
    double hte = x/y*yB/xB*0.5;
    double g3 = std::sqrt(yB*yB + hte*hte*4.*xB*xB);
    double g1 = std::sqrt(yT*yT + xT*xT);
    return Eigen::Vector<double,2>{hte,(r-g3)/(g1-g3)};
  }
};
struct circle_face_pullbacks : public Manicore::ParametrizedDerivedMap<2,2> 
{
  Eigen::Matrix<double,2,2> DI(Eigen::Vector<double,2> const & x_in) const override final {
    double xT = _extra[0];
    double xB = _extra[1];
    double yT = _extra[2];
    double yB = _extra[3];
    double ca = _extra[4];
    double sa = _extra[5];
    double t = x_in(0);
    double p = x_in(1);
    double rI = std::sqrt(yB*yB + t*t*4.*xB*xB);
    double R = std::sqrt(xT*xT+yT*yT);
    double tr = t*2.*xB/yB;
    double tmpI = (p*R + (1. - p)*rI)/std::sqrt(1.+tr*tr);
    double cthetap = -2.*tr*xB/yB/std::pow(1.+tr*tr,1.5);
    double rIp = 4.*t*xB*xB/rI;
    double Idt1 = (p*R + (1. -p)*rI)*cthetap + (1.-p)*rIp/std::sqrt(1.+tr*tr);
    double Idp1 = (R - rI)/std::sqrt(1+tr*tr);
    Eigen::Matrix<double,2,2> ref_val{{2.*xB/yB*tmpI + tr*Idt1, tr*Idp1 },
                                      {Idt1, Idp1}};
    return Eigen::Matrix<double,2,2>{{ca,-sa},{sa,ca}}*ref_val;
  }
  Eigen::Matrix<double,2,2> DJ(Eigen::Vector<double,2> const & x_in) const override final {
    double xT = _extra[0];
    double xB = _extra[1];
    double yT = _extra[2];
    double yB = _extra[3];
    double ca = _extra[4];
    double sa = _extra[5];
    Eigen::Vector<double,2> x_rot = Eigen::Matrix<double,2,2>{{ca,sa},{-sa,ca}}*x_in;
    double x = x_rot(0);
    double y = x_rot(1);
    double rJ = std::sqrt(x*x+y*y);
    double hteoverx = yB/y/xB*0.5;
    double hte = x*hteoverx;
    double g3 = std::sqrt(yB*yB + hte*hte*4.*xB*xB);
    double g1 = std::sqrt(yT*yT + xT*xT);
    double tmpJ = (rJ-g3)/(g1-g3);
    double rJx = x/rJ;
    double rJy = y/rJ;
    double g3x = x*yB*yB/y/y/g3;
    double g3y = -x*g3x/y;

    return Eigen::Matrix<double,2,2>{{hteoverx,-hte/y},
                                     {(rJx - g3x + tmpJ*g3x)/(g1-g3),
                                      (rJy - g3y + tmpJ*g3y)/(g1-g3)}}
           *Eigen::Matrix<double,2,2>{{ca,sa},{-sa,ca}};
  }
};

struct north_embedding : public Manicore::ParametrizedMap<3,2>
{
  Eigen::Vector<double,3> I(Eigen::Vector<double,2> const & x_in) const override final {
    double X = x_in(0);
    double Y = x_in(1);
    double r2 = X*X + Y*Y;
    return Eigen::Vector<double,3>{2.*X/(1.+r2),2.*Y/(1.+r2),(1. - r2)/(1. + r2)};
  }
  Eigen::Vector<double,2> J(Eigen::Vector<double,3> const & x_in) const override final {
    return Eigen::Vector<double,2>{};
  }
};
struct south_embedding : public Manicore::ParametrizedMap<3,2>
{
  Eigen::Vector<double,3> I(Eigen::Vector<double,2> const & x_in) const override final {
    double X = x_in(0);
    double Y = x_in(1);
    double r2 = X*X + Y*Y;
    return Eigen::Vector<double,3>{2.*X/(1.+r2),2.*Y/(1.+r2),(r2 - 1.)/(1. + r2)};
  }
  Eigen::Vector<double,2> J(Eigen::Vector<double,3> const & x_in) const override final {
    return Eigen::Vector<double,2>{};
  }
};

struct north_metric : public Manicore::ParametrizedMetricMap<2>
{
  north_metric() : Manicore::ParametrizedMetricMap<2>(1) {;}
  Eigen::Matrix<double,2,2> metric_inv(Eigen::Vector<double,2> const & x_in) const override final {
    double X = x_in(0);
    double Y = x_in(1);
    double r2 = X*X + Y*Y;
    constexpr double R2 = 1.;
    double tmp = (r2+R2)*(r2+R2)/(4.*R2);
    return Eigen::Matrix2d::Identity()*tmp;
  }
  Eigen::Matrix<double,2,2> metric(Eigen::Vector<double,2> const & x_in) const override final {
    double X = x_in(0);
    double Y = x_in(1);
    double r2 = X*X + Y*Y;
    constexpr double R2 = 1.;
    double tmp = 4.*R2/((r2+R2)*(r2+R2));
    return Eigen::Matrix2d::Identity()*tmp;
  }
  double volume(Eigen::Vector<double,2> const & x_in) const override final {
    double X = x_in(0);
    double Y = x_in(1);
    double r2 = X*X + Y*Y;
    constexpr double R2 = 1.;
    return 4.*R2/((r2+R2)*(r2+R2));
  }
};
struct south_metric : public Manicore::ParametrizedMetricMap<2>
{
  south_metric() : Manicore::ParametrizedMetricMap<2>(-1) {;}
  Eigen::Matrix<double,2,2> metric_inv(Eigen::Vector<double,2> const & x_in) const override final {
    double X = x_in(0);
    double Y = x_in(1);
    double r2 = X*X + Y*Y;
    constexpr double R2 = 1.;
    double tmp = (r2+R2)*(r2+R2)/(4.*R2);
    return Eigen::Matrix2d::Identity()*tmp;
  }
  Eigen::Matrix<double,2,2> metric(Eigen::Vector<double,2> const & x_in) const override final {
    double X = x_in(0);
    double Y = x_in(1);
    double r2 = X*X + Y*Y;
    constexpr double R2 = 1.;
    double tmp = 4.*R2/((r2+R2)*(r2+R2));
    return Eigen::Matrix2d::Identity()*tmp;
  }
  double volume(Eigen::Vector<double,2> const & x_in) const override final {
    double X = x_in(0);
    double Y = x_in(1);
    double r2 = X*X + Y*Y;
    constexpr double R2 = 1.;
    return 4.*R2/((r2+R2)*(r2+R2));
  }
};

Manicore::ParametrizedMap<3,2>* List_embedding_2to3(size_t id) {
  switch(id) {
    case(0):
      return new north_embedding;
    case(1):
      return new south_embedding;
    default:
      throw std::runtime_error("Unexpected embbeding id");
      return nullptr;
  }
}

Manicore::ParametrizedMetricMap<2>* List_metrics_2D(size_t id) {
  switch(id) {
    case(0):
      return new north_metric;
    case(1):
      return new south_metric;
    default:
      throw std::runtime_error("Unexpected map id");
      return nullptr;
  }
}

Manicore::ParametrizedMap<2,1>* List_edge_maps_2D(size_t id) {
  switch(id) {
    case(1):
      return new circle_edge_map;
    default:
      throw std::runtime_error("Unexpected map id");
      return nullptr;
  }
}

Manicore::ParametrizedMap<2,2>* List_face_maps_2D(size_t id) {
  switch(id) {
    case(1):
      return new circle_face_map;
    default:
      throw std::runtime_error("Unexpected map id");
      return nullptr;
  }
}

Manicore::ParametrizedDerivedMap<2,1>* List_edge_pullbacks_2D(size_t id) {
  switch(id) {
    case(1):
      return new circle_edge_pullbacks;
    default:
      throw std::runtime_error("Unexpected map id");
      return nullptr;
  }
}

Manicore::ParametrizedDerivedMap<2,2>* List_face_pullbacks_2D(size_t id) {
  switch(id) {
    case(1):
      return new circle_face_pullbacks;
    default:
      throw std::runtime_error("Unexpected map id");
      return nullptr;
  }
}

