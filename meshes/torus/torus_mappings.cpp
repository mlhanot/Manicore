#include "map_interface.hpp"


struct edge_map : public Manicore::ParametrizedMap<2,1> 
{
  Eigen::Vector<double,2> I(Eigen::Vector<double,1> const & t_v) const override final {
    double t = t_v(0);
    Eigen::Vector<double,2> V0{_extra[0],_extra[1]};
    Eigen::Vector<double,2> V1{_extra[2],_extra[3]};
    return t*V1 + (1. - t)*V0;
  }
  Eigen::Vector<double,1> J(Eigen::Vector<double,2> const & x) const override final {
    Eigen::Vector<double,2> V0{_extra[0],_extra[1]};
    Eigen::Vector<double,2> V1{_extra[2],_extra[3]};
    double hE2 = (V1 - V0).dot(V1 - V0);
    return Eigen::Vector<double,1>{(x-V0).dot(V1 - V0)/hE2};
  }
};
struct edge_pullbacks : public Manicore::ParametrizedDerivedMap<2,1> 
{
  Eigen::Matrix<double,2,1> DI(Eigen::Vector<double,1> const & x_in) const override final {
    Eigen::Vector<double,2> V0{_extra[0],_extra[1]};
    Eigen::Vector<double,2> V1{_extra[2],_extra[3]};
    return V1 - V0;
  }
  Eigen::Matrix<double,1,2> DJ(Eigen::Vector<double,2> const & x_in) const override final {
    Eigen::Vector<double,2> V0{_extra[0],_extra[1]};
    Eigen::Vector<double,2> V1{_extra[2],_extra[3]};
    double hE2 = (V1 - V0).dot(V1 - V0);
    return (V1 - V0).transpose()/hE2;
  }
};

struct face_map : public Manicore::ParametrizedMap<2,2> 
{
  Eigen::Vector<double,2> I(Eigen::Vector<double,2> const & x_in) const override final {
    Eigen::Vector<double,2> offset{_extra[0],_extra[1]};
    double scale = _extra[2];
    return scale*x_in + offset;
  }
  Eigen::Vector<double,2> J(Eigen::Vector<double,2> const & x_in) const override final {
    Eigen::Vector<double,2> offset{_extra[0],_extra[1]};
    double scale = _extra[2];
    return (x_in - offset)/scale;
  }
};
struct face_pullbacks : public Manicore::ParametrizedDerivedMap<2,2> 
{
  Eigen::Matrix<double,2,2> DI(Eigen::Vector<double,2> const & x_in) const override final {
    return Eigen::Matrix<double,2,2>::Identity()*_extra[2];
  }
  Eigen::Matrix<double,2,2> DJ(Eigen::Vector<double,2> const & x_in) const override final {
    return Eigen::Matrix<double,2,2>::Identity()/_extra[2];
  }
};

#ifdef EMBEDDING_3D
// TODO Test this
struct main_embedding : public Manicore::ParametrizedMap<3,2>
{
  Eigen::Vector<double,3> I(Eigen::Vector<double,2> const & x_in) const override final {
    double phi = x_in(0)*2.*std::numbers::pi;
    double theta = x_in(1)*2.*std::numbers::pi;
    Eigen::Vector<double,3> loc{2.+std::cos(theta),0.,std::sin(theta)};
    return Eigen::AngleAxisd(phi,Eigen::Vector3d::UnitZ())*loc;
  }
  Eigen::Vector<double,2> J(Eigen::Vector<double,3> const & x_in) const override final {
    return Eigen::Vector<double,2>{};
  }
};
struct main_pullback : public Manicore::ParametrizedDerivedMap<3,2>
{
  Eigen::Matrix<double,3,2> DI(Eigen::Vector<double,2> const & x_in) const override final {
    double phi = x_in(0)*2.*std::numbers::pi;
    double theta = x_in(1)*2.*std::numbers::pi;
    Eigen::Vector<double,3> loc{2.+std::cos(theta),0.,std::sin(theta)};
    Eigen::Vector<double,3> dloc{-std::sin(theta),0.,std::cos(theta)};
    Eigen::Matrix<double,3,2> rv;
    rv.col(0) = Eigen::AngleAxisd(phi+0.5*std::numbers::pi,Eigen::Vector3d::UnitZ())*loc;
    rv.col(1) = Eigen::AngleAxisd(phi,Eigen::Vector3d::UnitZ())*dloc;
    return rv;
  }
  Eigen::Matrix<double,2,3> DJ(Eigen::Vector<double,3> const & x_in) const override final {
    return Eigen::Matrix<double,2,3>{};
  }
};
#else
struct main_embedding : public Manicore::ParametrizedMap<3,2>
{
  Eigen::Vector<double,3> I(Eigen::Vector<double,2> const & x_in) const override final {
    return Eigen::Vector<double,3>{x_in(0),x_in(1),0.};
  }
  Eigen::Vector<double,2> J(Eigen::Vector<double,3> const & x_in) const override final {
    return Eigen::Vector<double,2>{};
  }
};
struct main_pullback : public Manicore::ParametrizedDerivedMap<3,2>
{
  Eigen::Matrix<double,3,2> DI(Eigen::Vector<double,2> const & x_in) const override final {
    Eigen::Matrix<double,3,2> rv = Eigen::Matrix<double,3,2>::Zero();
    rv(0,0) = 1.;
    rv(1,1) = 1.;
    return rv;
  }
  Eigen::Matrix<double,2,3> DJ(Eigen::Vector<double,3> const & x_in) const override final {
    return Eigen::Matrix<double,2,3>{};
  }
};
#endif

struct flat_metric : public Manicore::ParametrizedMetricMap<2>
{
  flat_metric() : Manicore::ParametrizedMetricMap<2>(1) {;}
  Eigen::Matrix<double,2,2> metric_inv(Eigen::Vector<double,2> const & x_in) const override final {
    return Eigen::Matrix2d::Identity();
  }
  Eigen::Matrix<double,2,2> metric(Eigen::Vector<double,2> const & x_in) const override final {
    return Eigen::Matrix2d::Identity();
  }
  double volume(Eigen::Vector<double,2> const & x_in) const override final {
    return 1;
  }
};

Manicore::ParametrizedMap<3,2>* List_embedding_2to3(size_t id) {
  return new main_embedding;
}

Manicore::ParametrizedDerivedMap<3,2>* List_pullback_2to3(size_t id) {
  return new main_pullback;
}

Manicore::ParametrizedMetricMap<2>* List_metrics_2D(size_t id) {
  return new flat_metric;
}

Manicore::ParametrizedMap<2,1>* List_edge_maps_2D(size_t id) {
  return new edge_map;
}

Manicore::ParametrizedMap<2,2>* List_face_maps_2D(size_t id) {
  return new face_map;
}

Manicore::ParametrizedDerivedMap<2,1>* List_edge_pullbacks_2D(size_t id) {
  return new edge_pullbacks;
}

Manicore::ParametrizedDerivedMap<2,2>* List_face_pullbacks_2D(size_t id) {
  return new face_pullbacks;
}

