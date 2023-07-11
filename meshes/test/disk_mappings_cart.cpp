#include "map_interface.hpp"


struct circle_edge_map : public Manicore::ParametrizedMap<2,1> 
{
  Eigen::Vector<double,2> I(Eigen::Vector<double,1> const & t) const override final {
    double xT = _extra[0];
    double yT = _extra[1];
    double ca = _extra[2];
    double sa = _extra[3];
    double xT2 = xT*xT;
    double yT2 = yT*yT;
    double l = std::sqrt((xT2+yT2)/(xT2*4.*t(0)*t(0) + yT2));
    Eigen::Vector<double,2> ref_val{xT*l*2.*t(0),yT*l};
    return Eigen::Matrix<double,2,2>{{ca,-sa},{sa,ca}}*ref_val;
  }
  Eigen::Vector<double,1> J(Eigen::Vector<double,2> const & x) const override final {
    double xT = _extra[0];
    double yT = _extra[1];
    double ca = _extra[2];
    double sa = _extra[3];
    Eigen::Vector<double,2> x_rot = Eigen::Matrix<double,2,2>{{ca,sa},{-sa,ca}}*x;
    double X = x_rot(0);
    double Y = x_rot(1);
    return Eigen::Vector<double,1>{0.5*(X/Y)*(yT/xT)};
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
    double x = x_in(0);
    double y = x_in(1);
    double xT2 = xT*xT;
    double yT2 = yT*yT;
    double R = std::sqrt(xT2+yT2);
    double Xy = xB*(yT-y)+xT*(y-yB);
    double l = (R*Xy/std::sqrt((x*xT*(yT-yB))*(x*xT*(yT-yB))+yT2*Xy*Xy)*(y - yB) + yT-y)/(yT-yB);
    Eigen::Vector<double,2> ref_val{l*x,l*y};
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
};
struct south_embedding : public Manicore::ParametrizedMap<3,2>
{
  Eigen::Vector<double,3> I(Eigen::Vector<double,2> const & x_in) const override final {
    double X = x_in(0);
    double Y = x_in(1);
    double r2 = X*X + Y*Y;
    return Eigen::Vector<double,3>{2.*X/(1.+r2),2.*Y/(1.+r2),(r2 - 1.)/(1. + r2)};
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

