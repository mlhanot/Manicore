#ifndef DEFINITIONS_HPP_INCLUDED
#define DEFINITIONS_HPP_INCLUDED

#include <Eigen/Dense>

#include <array>

/** \file definitions.hpp
  Reference for signature of the functions that must be provided alongside the mesh.
  */

namespace Manicore {
  
  /// Array of d+1 points of \f$\mathbb{R}^d\f$
  template<size_t d>
    using Simplex = std::array<Eigen::Vector<double,d>,d+1>;
  
  /// Used for the parametrization of the mesh elements
  /** \tparam dimension %Dimension of the manifold
    \tparam d %Dimension of the cell 
    */
  template<size_t dimension,size_t d>
    struct ParametrizedMap {
      /// Parametrization from a reference element to a chart
      virtual Eigen::Vector<double,dimension> I(Eigen::Vector<double,d> const &) const = 0;
      /// Inverse mapping from a chart to the reference element
      virtual Eigen::Vector<double,d> J(Eigen::Vector<double,dimension> const &) const = 0;
      /// Optional parameters that may be used within the class
      std::vector<double> _extra;
      virtual ~ParametrizedMap() = default;
    };

  /// First order differentials of the parametrizations
  /** \tparam d1 %Dimension of the manifold
    \tparam d2 %Dimension of the cell 
    */
  template<size_t d1,size_t d2>
    struct ParametrizedDerivedMap {
      /// Differential of the parametrization
      /** From the (tangent space of) the reference element to the (tangent space of) the chart */
      virtual Eigen::Matrix<double,d1,d2> DI(Eigen::Vector<double,d2> const &) const = 0;
      /// Differential of the inverse mapping
      /** From the (tangent space of) the reference element to the (tangent space of) the chart */
      virtual Eigen::Matrix<double,d2,d1> DJ(Eigen::Vector<double,d1> const &) const = 0;
      /// Optional parameters that may be used within the class
      std::vector<double> _extra;
      virtual ~ParametrizedDerivedMap() = default;
    };

  /// Used to specify the ambient metric
  /** \tparam dimension %Dimension of the manifold
    */
  template<size_t dimension>
    struct ParametrizedMetricMap {
      /// Metric of the tangent space on a chart
      virtual Eigen::Matrix<double,dimension,dimension> metric(Eigen::Vector<double,dimension> const &) const = 0;
      /// Metric of the cotangent space on a chart
      /** This is the inverse matrix of the metric matrix */
      virtual Eigen::Matrix<double,dimension,dimension> metric_inv(Eigen::Vector<double,dimension> const &) const = 0;
      /// Scaling of the volume form on a chart
      /** This is the determinant of the inverse of the metric */
      virtual double volume(Eigen::Vector<double,dimension> const &) const = 0;
      /// Optional parameters that may be used within the class
      std::vector<double> _extra;
      /// Allows setting the orientation when sub classing
      ParametrizedMetricMap(int o) : orientation(o) {;}
      /// Orientation of the chart 
      /** Should \f$ dx^1 \wedge dx^2 \wedge \dots \wedge dx^n \f$ be a direct basis ? */
      const int orientation = 0;
      virtual ~ParametrizedMetricMap() = default;
    };

  /// Do nothing mapping (always return null)
  template<size_t dimension,size_t d>
    ParametrizedMap<dimension,d> *DefaultMapping(size_t id) {return nullptr;}

} // namespace

#endif

