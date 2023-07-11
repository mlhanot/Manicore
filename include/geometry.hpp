#ifndef GEOMETRY_HPP_INCLUDED
#define GEOMETRY_HPP_INCLUDED

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "definitions.hpp"

/** \file geometry.hpp
  Helpers to compute geometric quantities.
  */
namespace Manicore::Geometry {

  /// Volume of a tetrahedron in 3D given by its 4 vertices
  inline double volume_tetrahedron(Eigen::Vector3d const & a, 
                            Eigen::Vector3d const & b, 
                            Eigen::Vector3d const & c, 
                            Eigen::Vector3d const & d) {
    return std::abs((a-d).dot((b-d).cross(c-d)))/6.;

  } 
  /// Area of a triangle in 3D given by its 3 vertices
  inline double volume_triangle(Eigen::Vector3d const &x1,
                         Eigen::Vector3d const &x2, 
                         Eigen::Vector3d const &x3) {
    double a = (x1 - x2).squaredNorm();
    double b = (x1 - x3).squaredNorm();
    double c = (x2 - x3).squaredNorm();
    // Heron(s formula
    return std::sqrt((a+b+c)*(a+b+c) - 2.*(a*a+b*b+c*c))*0.25;
  }
  /// Area of a triangle in 2D given by its 3 vertices
  inline double volume_triangle(Eigen::Vector2d const &x1,
                         Eigen::Vector2d const &x2,
                         Eigen::Vector2d const &x3) {
    return std::abs((x1(0)-x3(0))*(x2(1)-x1(1)) - (x1(0)-x2(0))*(x3(1)-x1(1)))*0.5;
  }

  /// Diameter of the convex hull of a vector of vertices in ND
  template<typename T>
  double diameter(std::vector<T> const & vl) {
    double diam = 0.;
    for (size_t i = 0; i < vl.size(); ++i) {
      for (size_t j = 0; j < vl.size(); ++j) {
        diam = std::max(diam,(vl[i]-vl[j]).squaredNorm());
      }
    }
    return std::sqrt(diam);
  }
  /// Compute a possible basis for the tangent space from a vector of vertices.
  /** Generate a set of vector spanning the basis by subtracting 2 vertices, and apply a Gram-Schmidt process
    */
  template<size_t dimension,size_t d>
  Eigen::Matrix<double,dimension,d> tangent_space(std::vector<Eigen::Vector<double,dimension>> const & vl) {
    assert(vl.size() > d);
    assert(d>1);
    Eigen::Matrix<double,dimension,d> rv;
    rv.col(0) = (vl[1]-vl[0]).normalise();
    size_t nb_filled = 1;
    for (size_t j = 1; j < vl.size(); ++j) {
      Eigen::Vector<double,dimension> tmp = vl[j]-vl[0];
      for (size_t i = 0; i < nb_filled; ++i) {
        double dotv = tmp.dot(rv.col(i));
        tmp = tmp - dotv*rv.col(i);
      }
      if (tmp.squaredNorm() > 1e-10) {
        rv.col(nb_filled) = tmp.normalise();
        nb_filled++;
        if (nb_filled == d) return rv;
      }
    }
    return rv;
  }

  /// Compute the centroid of a simplex S
  template<size_t d> 
  Eigen::Vector<double,d> middleSimplex(Simplex<d> const & S)
  {
    Eigen::Vector<double,d> mid = S[d];
    for (size_t i = 0; i < d; ++i) {
      mid += S[i];
    }
    return mid/(d+1.);
  }

  /// Is a vector x inside a simplex S
  template<size_t d> 
  bool inside(Eigen::Vector<double,d> const &x, Simplex<d> const & S)
  {
    constexpr double esp = 1e-10;
    Eigen::Matrix<double,d,d> T;
    for (size_t i = 0; i < d; ++i) {
      for (size_t j = 0; j < d; ++j) {
        T(i,j) = S[j](i) - S[d](i);
      }
    }
    Eigen::Vector<double,d> L = T.partialPivLu().solve(x - S[d]);
    double p_sum = 0.;
    for (size_t i = 0; i < d; ++i) {
      if (L(i) < -esp || L(i) > 1. + esp) return false;
      p_sum += L(i);
    }
    return p_sum < 1. + esp && p_sum > -esp;
  }

} // end namespace Volume

#endif

