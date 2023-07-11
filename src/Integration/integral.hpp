/* 
 * Copyright (c) 2023 Marien Hanot <marien-lorenzo.hanot@umontpellier.fr>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *  
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef INTEGRAL_HPP_INCLUDED
#define INTEGRAL_HPP_INCLUDED

#include "quadraturerule.hpp"
#include "mesh.hpp"

/** @defgroup Integration
  @brief Classes providing support to compute mass matrices
  */

namespace Manicore {

  /** \file integral.hpp
    Generate quadrature and evaluate them
    */

  /// \addtogroup Integration
  ///@{

  // TODO add a cache
  // Ensure that mesh survives this class
  /// Interface with quadrature rule
  /** \tparam dimension %Dimension of the manifold
    \tparam d %Dimension of the cell
    \warning This does not take ownership of the mesh but keep a pointer of it. This class is intended to be short-lived but ensure that the mesh survives this class.

    Generate the quadrature rule and implement the evaluate of every quantity used.
    */
  template<size_t dimension,size_t d> requires(d > 0 && d <= dimension)
  class Integral {
    public:
      /** \warning Ensure that the mesh survives this class */
      Integral(const Mesh<dimension>* mesh /*!< Mesh used for the quadrature*/) : _mesh(mesh) {;}
      
      /// Generic interface to generate quadrature rule on any dimension
      QuadratureRule<d> generate_quad(size_t i_cell /*!< Cell index */, int dqr /*!< Degree of exactness */) const 
      {
        return generate_quadrature_rule(_mesh->template get_cell_map<d>(i_cell),dqr);
      }

      /// Evaluate the polynomial basis on a cell
      /** \return Matrix of size : [quad.size(), Dimension::PolyDim (r,d) ] 
       The columns correspond to the polynomial basis */
      Eigen::MatrixXd evaluate_scalar_quad(size_t i_cell /*!< Cell index */,
                                           int r /*!< Polynomial degree */,
                                           QuadratureRule<d> const & quad /*!< Quadrature rule */) const;
      /// Evaluate the polynomial basis on a quadrature of the boundary
      /** \return Matrix of size : [quad.size(), Dimension::PolyDim (r,d) ] 
       The columns correspond to the polynomial basis */
      Eigen::MatrixXd evaluate_scalar_quad_tr(size_t i_cell /*!< Cell index */,
                                              size_t bd_rel_index /*!< Relative index of the boundary (e.g. between 0 and 2 for a triangle) */,
                                              int r /*!< Polynomial degree */,
                                              QuadratureRule<d-1> const & quad /*!< Quadrature rule on the boundary */) const requires(d > 1);

      /// Evaluate the volume form on a quadrature of the boundary
      /** \return Vector of size : quad.size() */
      Eigen::VectorXd evaluate_volume_form(size_t i_cell /*!< Cell index */,QuadratureRule<d> const & quad /*!< Quadrature rule */) const;

      /// Evaluate the pullback by I of the \f$L^2\f$ product on the exterior algebra of the exterior algebra on the reference element.
      /** Compute the matrix of \f$ I_T^* (\langle \cdot , \cdot \rangle_{g}) \f$.
        \tparam l Form degree
        */
      template<size_t l>
      std::vector<Eigen::Matrix<double,Dimension::ExtDim(l,d),Dimension::ExtDim(l,d)>> 
              evaluate_exterior_quad(size_t i_cell /*!< Cell index */,
                                     QuadratureRule<d> const & quad /*!< Quadrature rule */) const;

      /// Evaluate the pullback by I of the \f$L^2\f$ product on the exterior algebra of the exterior algebra on the reference element composed with the trace on the right.
      /** Compute the matrix of \f$ I_F^* (\langle \cdot , I_F^* J_T^* \cdot \rangle_{g}) \f$.
        \tparam l Form degree
        */
      template<size_t l>
      std::vector<Eigen::Matrix<double,Dimension::ExtDim(l,d-1),Dimension::ExtDim(l,d)>> 
              evaluate_exterior_quad_tr(size_t i_cell /*!< Cell index */,
                                        size_t bd_rel_index /*!< Relative index of the boundary */, 
                                        QuadratureRule<d-1> const & quad /*!< Quadrature rule on the boundary */) const requires(d > 1);

      /// Access the mesh associated with this object
      const Mesh<dimension>* mesh() const {return _mesh;}
    private:
      const Mesh<dimension>* _mesh; // Do not own the mesh
  };
  ///@}

  // ---------------------------------------------------------------------------------------------------------
  // Implementation
  // ---------------------------------------------------------------------------------------------------------
  template<size_t dimension,size_t d> requires(d > 0 && d <= dimension)
  Eigen::MatrixXd Integral<dimension,d>::evaluate_scalar_quad(
                                             size_t i_cell, // Object id
                                             int r, // Polynomial degree
                                             QuadratureRule<d> const & quad) const 
  {
    auto const & F = _mesh->template get_cell_map<d>(i_cell);
    const size_t nbp = quad.size();
    Eigen::MatrixXd rv(nbp,Dimension::PolyDim(r,d));

    for (size_t iqr = 0; iqr < nbp; ++iqr) {
      auto const x = quad[iqr].vector;
      for (size_t i_b = 0; i_b < Dimension::PolyDim(r,d); ++i_b) {
        rv(iqr,i_b) = F.evaluate_poly_on_ref(x,i_b,r);
      }
    }
    return rv;
  }

  template<size_t dimension,size_t d> requires(d > 0 && d <= dimension)
  Eigen::MatrixXd Integral<dimension,d>::evaluate_scalar_quad_tr(
                                             size_t i_cell, // Object id
                                             size_t bd_rel_index,
                                             int r, // Polynomial degree
                                             QuadratureRule<d-1> const & quad) const requires(d > 1)
  {
    auto const & T = _mesh->template get_cell_map<d>(i_cell);
    auto const & F = _mesh->template get_cell_map<d-1>(_mesh->get_boundary(d-1,d,i_cell)[bd_rel_index]);
    size_t bd_map = _mesh->get_relative_map(d-1,d,i_cell)[bd_rel_index];

    const size_t nbp = quad.size();
    Eigen::MatrixXd rv(nbp,Dimension::PolyDim(r,d));

    for (size_t iqr = 0; iqr < nbp; ++iqr) {
      auto const Ix = F.evaluate_I(bd_map,quad[iqr].vector);
      for (size_t i_b = 0; i_b < Dimension::PolyDim(r,d); ++i_b) {
        rv(iqr,i_b) = T.evaluate_poly_pullback(0,Ix,i_b,r);
      }
    }
    return rv;
  }


  template<size_t dimension,size_t d> requires(d > 0 && d <= dimension)
  Eigen::VectorXd Integral<dimension,d>::evaluate_volume_form(size_t i_cell,QuadratureRule<d> const & quad) const
  {
    auto const & F = _mesh->template get_cell_map<d>(i_cell);
    const size_t nbp = quad.size();
    Eigen::VectorXd rv(nbp);

    for (size_t iqr = 0; iqr < nbp; ++iqr) {
      auto const x = quad[iqr].vector;
      auto const DI = F.evaluate_DI(0,x);
      if constexpr(d == dimension) {
        rv(iqr) = _mesh->volume_form(_mesh->get_map_ids(d,i_cell)[0],F.evaluate_I(0,x))*DI.determinant();
      } else {
        rv(iqr) = std::sqrt((DI.transpose()*_mesh->metric(_mesh->get_map_ids(d,i_cell)[0],F.evaluate_I(0,x))*DI).determinant());
      }
    }
    return rv;
  }

  template<size_t dimension,size_t d> requires(d > 0 && d <= dimension)
  template<size_t l>
  std::vector<Eigen::Matrix<double,Dimension::ExtDim(l,d),Dimension::ExtDim(l,d)>> Integral<dimension,d>::evaluate_exterior_quad(size_t i_cell,QuadratureRule<d> const & quad) const 
  {
    auto const & F = _mesh->template get_cell_map<d>(i_cell);
    const size_t nbp = quad.size();
    std::vector<Eigen::Matrix<double,Dimension::ExtDim(l,d),Dimension::ExtDim(l,d)>> rv;
    rv.reserve(nbp);

    for (size_t iqr = 0; iqr < nbp; ++iqr) {
      auto const x = quad[iqr].vector;
      auto const Ix = F.evaluate_I(0,x);

      if constexpr(dimension == d) { // TODO Check if this is actually faster
        auto const Jp = F.template evaluate_DJ_p<l>(0,Ix);
        rv.emplace_back(Jp.transpose()*Compute_ExtGram<l>::compute(_mesh->metric_inv(_mesh->get_map_ids(d,i_cell)[0],Ix))*Jp);
      } else { // Compute the pullback of the metric, and then inverse it. Can we avoid this computation?
        auto const DI = F.evaluate_DI(0,x);
        Eigen::Matrix<double,d,d> invIpG = (DI.transpose()*_mesh->metric(_mesh->get_map_ids(d,i_cell)[0],Ix)*DI).inverse();
        rv.emplace_back(Compute_ExtGram<l>::compute(invIpG));
      }
    } 

    return rv;
  }

  template<size_t dimension,size_t d> requires(d > 0 && d <= dimension)
  template<size_t l>
  std::vector<Eigen::Matrix<double,Dimension::ExtDim(l,d-1),Dimension::ExtDim(l,d)>> Integral<dimension,d>::evaluate_exterior_quad_tr(size_t i_cell,size_t bd_rel_index, QuadratureRule<d-1> const & quad) const requires(d > 1)
  {
    auto const & T = _mesh->template get_cell_map<d>(i_cell);
    auto const & F = _mesh->template get_cell_map<d-1>(_mesh->get_boundary(d-1,d,i_cell)[bd_rel_index]);
    size_t bd_map = _mesh->get_relative_map(d-1,d,i_cell)[bd_rel_index];

    const size_t nbp = quad.size();
    std::vector<Eigen::Matrix<double,Dimension::ExtDim(l,d-1),Dimension::ExtDim(l,d)>> rv;
    rv.reserve(nbp);

    for (size_t iqr = 0; iqr < nbp; ++iqr) {
      auto const x = quad[iqr].vector;
      auto const Ix = F.evaluate_I(bd_map,x);
      auto const DIF = F.evaluate_DI(bd_map,x);
      auto const IpF = F.template evaluate_DI_p<l>(bd_map,x);
      auto const JpT = T.template evaluate_DJ_p<l>(0,Ix);
      Eigen::Matrix<double,d-1,d-1> invIpG = (DIF.transpose()*_mesh->metric(_mesh->get_map_ids(d,i_cell)[0],Ix)*DIF).inverse();
      rv.emplace_back(Compute_ExtGram<l>::compute(invIpG)*IpF*JpT);
    }

    return rv;
  }
} // end namespace

#endif

