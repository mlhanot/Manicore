// Author: Daniele Di Pietro (daniele.di-pietro@umontpellier.fr)
// Author: Jérome Droniou (jerome.droniou@monash.edu)
//
// Adapted to Manicore by Marien Hanot

#ifndef QUADRATURERULE_HPP
#define QUADRATURERULE_HPP

#include "dcell.hpp"
#include "legendregauss.hpp"
#include "quad_2d.hpp"

/** \file quadraturerule.hpp 
  Wrapper to provide an uniform interface to every quadrature rule
  */

/*!  
*  @defgroup Quadratures 
* @brief Classes providing quadratures on edges and in cells
*/

namespace Manicore {
  /// \addtogroup Quadratures
  ///@{

  /// Maximum degree of quadrature implemented for each dimension (start with dimension 1)
  constexpr int QuadratureMaxDegree[] = {21*2,20};

  /// Description of one node and one weight from a quadrature rule
  /** \tparam d %Dimension of the cell */
  template<size_t d>
  struct QuadratureNode 
  {
    Eigen::Vector<double,d> vector /*! Location on the reference element */;
    double w /*! Weight */;
  };

  /// Vector of locations and weights
  /** \tparam d %Dimension of the cell */
  template<size_t d>
    using QuadratureRule = std::vector<QuadratureNode<d>>;

  /// Generate a quadrature rule for the cell f 
  /** Specialization for the dimension 1 */
  template<typename CellType> requires(CellType::cell_dim == 1)
  QuadratureRule<CellType::cell_dim> generate_quadrature_rule(const CellType &f /*!< dCell_map */ , const int doe /*!< Degree of exactness */)
  {
    auto const &T = f.get_reference_elem();
    QuadratureRule<1> quad;
    for (auto const & trig : T) {
      double v = trig[1](0) - trig[0](0);
      double x0 = trig[0](0);
      LegendreGauss rule(std::max(doe,1));
      for (size_t iqn = 0; iqn < rule.npts(); ++iqn) {
        Eigen::Vector<double,1> vect = Eigen::Vector<double,1>{rule.tq(iqn)*v + x0};
        quad.push_back({vect,rule.wq(iqn)});
      }
    }
    return quad;
  }

  /// Generate a quadrature rule for the cell f 
  /** Specialization for the dimension 2 */
  template<typename CellType> requires(CellType::cell_dim == 2)
  QuadratureRule<CellType::cell_dim> generate_quadrature_rule(const CellType &f /*!< dCell_map */ , const int doe /*!< Degree of exactness */)
  {
    QuadRuleTriangle quadCell(std::max(doe,0));
    QuadratureRule<2> quad;

    auto const &T = f.get_reference_elem();
    for (auto const & trig : T) {
      double xTt[] = {trig[0](0),trig[1](0),trig[2](0)};
      double yTt[] = {trig[0](1),trig[1](1),trig[2](1)};
      quadCell.setup(xTt, yTt);
      for (size_t iqn = 0; iqn < quadCell.nq(); iqn++) {
          quad.push_back({Eigen::Vector2d{quadCell.xq(iqn), quadCell.yq(iqn)}, quadCell.wq(iqn)});
      }
    }
    return quad;
  }

  ///@}
}  // end of namespace 
#endif
