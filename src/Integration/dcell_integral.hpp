#ifndef DCELL_INTEGRAL_HPP_INCLUDED
#define DCELL_INTEGRAL_HPP_INCLUDED

#include "integral.hpp"

/** \file dcell_integral.hpp
  Compute the mass matrices and traces operator of a cell
  */
namespace Manicore {
  /// \addtogroup Integration
  ///@{

  /// Compute the mass matrices of a d-cell
  /** \tparam dimension %Dimension of the manifold
    \tparam d %Dimension of the cell
    */
  template<size_t dimension,size_t d> requires(d > 0 && d <= dimension)
  struct dCell_mass {
      dCell_mass(size_t i_cell /*!< Cell index*/, int r /*!< Polynomial degree */, const QuadratureRule<d> & quad /*!< Quadrature rule to use */, const Integral<dimension,d> &integral /*!< Integral object generating the quadrature */);
      /// Do nothing
      dCell_mass() {;}

      /// Mass matrices of all form degree 
      /** masses[k] is the mass for the k-forms */
      std::array<Eigen::MatrixXd,dimension+1> masses;
  };

  /// Compute the traces matrices of a d-cell onto its boundary
  /** \tparam dimension %Dimension of the manifold
    \tparam d %Dimension of the cell
    */
  template<size_t dimension,size_t d> requires(d > 0 && d <= dimension)
  struct dCell_traces {
      dCell_traces(size_t i_cell /*!< Cell index */, int r /*!< Polynomial degree */, int dqr /*!< Quadrature degree */, 
                   const std::vector<dCell_mass<dimension,d-1>> & b_masses /*!< Masses of all the cell of dimension d-1 */,
                   const Integral<dimension,d> &integral /*!< Integral object for the dimension d */, 
                   const Integral<dimension,d-1> &integral_b /*!< Integral object for the dimension d-1 */);
      /// Do nothing
      dCell_traces() {;}

      // The array index is the form degree (0 to d-1), and the vector index is the boundary
      /// Trace matrices for all form degree and all boundary cell 
      /** traces[k][i_b] is the trace for a k-form on the i_b-boundary cell, using relative index 

      starTraces[k][i_b] is \f$ \star \text{tr} \star^{-1} \f$ of a \f$d-k\f$-form on the i_b-boundary cell, using relative index 
        
        The trace of d-form is not included. The array uses the global dimension to give an uniform interface, however only the first d elements are used.
       */
      std::array<std::vector<Eigen::MatrixXd>,dimension> traces;
      std::array<std::vector<Eigen::MatrixXd>,dimension> starTraces; // TODO: only used for the stabilization term in the L2-product, check if this is really necessary
  };
  /// Specialization for edges
  template<size_t dimension> 
  struct dCell_traces<dimension,1> {
      dCell_traces(size_t i_cell /*!< Edge index */, int r /*!< Polynomial degree */, const Mesh<dimension>* mesh /*!< Reference to the mesh*/);
      /// Do nothing
      dCell_traces() {;}

      // The array index is the form degree (0 to d-1), and the vector index is the boundary
      /// Trace matrices for all form degree and all boundary cell 
      /** traces[k][i_b] is the trace for a k-form on the i_b-th vertex, using relative index 
        
      starTraces[k][i_b] is \f$ \star \text{tr} \star^{-1} \f$ of a \f$d-k\f$-form on the i_b-boundary cell, using relative index 

        Only traces[0] is used.
       */
      std::array<std::vector<Eigen::MatrixXd>,dimension> traces;
      std::array<std::vector<Eigen::MatrixXd>,dimension> starTraces;
  };
  ///@}
}

#endif

