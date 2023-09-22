#ifndef EXPORTED_HPP
#define EXPORTED_HPP

#include <Eigen/Dense>
#include <vector>

#include "exterior_dimension.hpp"
#include "mesh.hpp"

/**
  @defgroup Helpers
  @brief Tools to perform various task
*/
/** \file exporter.hpp
  Export discrete objects to be display in external software
*/

namespace Manicore {
  /// \addtogroup Helpers
  ///@{

  /// Evaluate vector of discrete unknowns and store them in files
  /** This class compute and store all the necessary data when constructed.

    \tparam dimension %Dimension of the manifold
    */
  template<size_t dimension>
  class Exporter {
    public:
      /// Constructor
      /** The constructor store the mapping between the cell and the embedding at each point.
        This may requires a significant amount of space.
        The location chosen to print the value are computed using the quadrature formulas. 
        The degree acc, is not directly the number of location per cell, but the degree of the formula used.

        The distribution may not be uniform, and may depends of the triangulation of the reference element in each cell.
        */
      Exporter(Mesh<dimension> const * mesh /*!< Reference to the mesh used */,
               int r /*!< Polynomial degree */,
               int acc /*!< Degree of the quadrature */
              );

      /// Write a csv file with the point-wise value of the discrete function
      /**
        By default, this use the identification between 0-forms and function, and between 1-forms and vector.
        In 2D, there is an ambiguity between \f$1\f$ and \f$n-1\f$ forms. 
        By default, this function will print an \f$1\f$-form, set star=true to reverse the behavior.

        \remark The DDR_Spaces::potential() operator returns the Hodge star of the potential
        */
      int save(size_t k /*!< Form degree */, 
                std::function<Eigen::VectorXd(size_t iT)> Fu_h /*!< Return the discrete vector in \f$\mathcal{P}_r\Lambda^k(\mathbb{R}^d)\f$ in a given cell iT */,
                const char *filename /*!< Name and location to write the data */,
                bool star=false /*!<Apply the Hodge star before exporting an 1-form */) const;
      /// Write a csv file with the point-wise value of the contraction of the function with itself
      /**
        Compute \f$ u^i u_i \f$
        */
      int saveSq(size_t k /*!< Form degree */, 
                    std::function<Eigen::VectorXd(size_t iT)> Fu_h /*!< Return the discrete vector in \f$\mathcal{P}_r\Lambda^k(\mathbb{R}^d)\f$ in a given cell iT */,
                    const char *filename /*!< Name and location to write the data */) const;

    private:
      std::vector<Eigen::Vector3d> _locations;
      std::vector<size_t> _cellIndices;
      std::vector<Eigen::Matrix<double,dimension,dimension>> _metricInv;
      std::vector<double> _volume;
      std::vector<Eigen::VectorXd> _polyEval;
      // TODO Make it works in arbitrary dimension. What to do with forms not match a scalar or a vector?
      std::vector<Eigen::Matrix<double,Dimension::ExtDim(1,3),Dimension::ExtDim(1,dimension)>> _pushforward;
      std::vector<Eigen::Matrix<double,Dimension::ExtDim(1,3),Dimension::ExtDim(dimension-1,dimension)>> _pushforwardStar;
      /* // Structure to store various size of pullback matrices
      template<size_t _k> struct PullbackHolder {
        std::vector<Eigen::Matrix<double,Dimension::ExtDim(_k,3),Dimension::ExtDim(_k,dimension)>> _pullback;
        PullbackHolder<_k-1> _next;
        template<size_t k> std::vector<Eigen::Matrix<double,Dimension::ExtDim(k,3),Dimension::ExtDim(k,dimension)>> & pullback() {
          if constexpr (k==_k) return _pullback;
          else return _next.template pullback<k>();
        }
        template<size_t k> std::vector<Eigen::Matrix<double,Dimension::ExtDim(k,3),Dimension::ExtDim(k,dimension)>> pullback() const {
          if constexpr (k==_k) return _pullback;
          else return _next.template pullback<k>();
        }
      };
      template<> struct PullbackHolder<0> {
        std::vector<Eigen::Matrix<double,1,1>> _pullback;
        template<size_t k> std::vector<Eigen::Matrix<double,1,1>> & pullback() {
          static_assert(k == 0);
          return _pullback;
        }
        template<size_t k> std::vector<Eigen::Matrix<double,1,1>> pullback() const {
          static_assert(k == 0);
          return _pullback;
        }
      };
      PullbackHolder<dimension> _pullbackEval;
      */ 
  };
  ///@}
}
#endif
