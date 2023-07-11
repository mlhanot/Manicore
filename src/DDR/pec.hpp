#ifndef PEC_HPP
#define PEC_HPP

#include "dcell_integral.hpp"

#include <iostream>

/** \file pec.hpp
  Discrete spaces for DDR-PEC
*/
/**
  @defgroup DDR
  @brief Classes implementing the DDR-PEC method on manifolds
*/

namespace Manicore {

  /// \addtogroup DDR
  ///@{

  /// Implement the discrete spaces of DDR-PEC
  /** This class compute and store the matrices of the differential operator, the Koszul operator, the trimmed polynomial basis, 
    and the masses and traces of every cell.

    The matrices for the differential, Koszul and trimmed basis are global and shared for all cells.
    The mass matrices and traces are computed and stored for each cell.

    The basis used for the operator are given in exterior_algebra.hpp

    \tparam dimension %Dimension of the manifold
    */
  template<size_t dimension> 
  class PEC {
    public:
      PEC(Mesh<dimension> const & mesh, /*!< Mesh to use */
                 int r, /*!< Polynomial degree */
                 bool use_threads = true, /*!< Enable pthreads parallelism */
                 std::array<int,dimension> const * dqr_p = nullptr, /*!< Degree of quadrature used to generate the mass matrices. It cannot be exact for generic metric and default to a pretty high degree. Use a lower degree if the metric and cells are almost flat. */
                 std::ostream & output = std::cout /*!< Output stream for status messages. */
                 );


      /// Return the mass matrix for the k-forms on the i-th d-cell in the basis PL(r,k,d)
      /** \return Mass matrix in the basis \f$ P_r\Lambda^k(\mathbb{R}^d)\f$ */
      Eigen::MatrixXd get_mass(size_t k /*!<Form degree*/, size_t d /*!< Cell dimension */,size_t i_cell /*!< Cell index */) const; 
      /// Return the trace for the k-forms on the i-th d-cell onto its j-th (d-1)-neighbour
      /** \return Trace matrix in the basis \f$ P_r\Lambda^k(\mathbb{R}^d) \rightarrow P_r\Lambda^k(\mathbb{R}^{d-1}) \f$ */
      Eigen::MatrixXd get_trace(size_t k /*!<Form degree*/, size_t d /*!< Cell dimension */,size_t i_cell /*!< Cell index */, size_t j_bd /*!< Relative index of the boundary element (e.g. between 0 and 2 for a triangle)*/) const; 

      // Getter for the generic operators matrices
      /// Return the image of the differential operator
      /** \return \f$ P_r\Lambda^l(\mathbb{R}^d) \rightarrow P_{r-1}\Lambda^{l+1}(\mathbb{R}^{d}) \f$ */
      const Eigen::MatrixXd & get_diff(size_t l /*!<Form degree*/, size_t d /*!<%Dimension*/) const {return _list_diff[_cmp_ind(l,d)];}
      /// Return the image of the Koszul operator
      /** \return \f$ P_r\Lambda^l(\mathbb{R}^d) \rightarrow P_{r+1}\Lambda^{l-1}(\mathbb{R}^{d}) \f$ */
      const Eigen::MatrixXd & get_Koszul(size_t l /*!<Form degree*/, size_t d /*!<%Dimension*/) const {return _list_Koszul[_cmp_ind(l,d)];}
      /// Return the image of the differential operator
      /** \return \f$ P_r\Lambda^l(\mathbb{R}^d) \rightarrow P_{r}\Lambda^{l+1}(\mathbb{R}^{d}) \f$ */
      const Eigen::MatrixXd & get_diff_as_degr(size_t l /*!<Form degree*/, size_t d /*!<%Dimension*/) const {return _list_diff_as_degr[_cmp_ind(l,d)];}
      /// Return the image of the trimmed polynomial basis
      /** \return \f$ P_r^{-}\Lambda^l(\mathbb{R}^d) \rightarrow P_{r}\Lambda^{l}(\mathbb{R}^{d}) \f$ */
      const Eigen::MatrixXd & get_trimmed(size_t l /*!<Form degree*/, size_t d /*!<%Dimension*/) const {return _list_trimmed[_cmp_ind(l,d)];}
      /// Return the image of the Koszul operator
      /** \return \f$ P_{-1}\Lambda^l(\mathbb{R}^d) \rightarrow P_{r}\Lambda^{l+1}(\mathbb{R}^{d}) \f$ */
      const Eigen::MatrixXd & get_reduced_Koszul_m1(size_t l /*!<Form degree*/, size_t d /*!<%Dimension*/) const {return _list_reduced_Koszul_m1[_cmp_ind(l,d)];}

    private:
      template<size_t d,size_t l> void _fill_lists();
      inline int _cmp_ind(size_t l, size_t d) const {return _dim_table[d-1]+l;}

      int _r;
      std::array<size_t,dimension+1> _dim_table;
      std::vector<Eigen::MatrixXd> _list_diff, // Image of dPL(r,l,d) inside PL(r-1,l+1,d)
                                   _list_Koszul, // Image of kPL(r,l,d) inside PL(r+1,l-1,d)
                                   _list_diff_as_degr, // Image of dPL(r,l,d) inside PL(r,l+1,d)
                                   _list_trimmed, // Image of PLtrimmed(r,l,d)
                                   _list_reduced_Koszul_m1; // Image of kPL(r-1,l,d)

      template<size_t _dimension,size_t _d>
      class dCellVariableList {
        private:
          std::vector<dCell_mass<_dimension,_d>> _mass;
          std::vector<dCell_traces<_dimension,_d>> _traces;
          dCellVariableList<_dimension,_d-1> _dCVL;
        public:
          template<size_t d> std::vector<dCell_mass<_dimension,d>> & mass() {
            if constexpr(d==_d) return _mass; 
            else return _dCVL.template mass<d>();
          }
          template<size_t d> std::vector<dCell_mass<_dimension,d>> mass() const {
            if constexpr(d==_d) return _mass; 
            else return _dCVL.template mass<d>();
          }
          template<size_t d> std::vector<dCell_traces<_dimension,d>> & traces() {
            if constexpr(d==_d) return _traces; 
            else return _dCVL.template traces<d>();
          }
          template<size_t d> std::vector<dCell_traces<_dimension,d>> traces() const {
            if constexpr(d==_d) return _traces; 
            else return _dCVL.template traces<d>();
          }
      };

      template<size_t _dimension>
      class dCellVariableList<_dimension,1> {
        private:
          std::vector<dCell_mass<_dimension,1>> _mass;
          std::vector<dCell_traces<_dimension,1>> _traces;
        public:
          template<size_t d> std::vector<dCell_mass<_dimension,d>> & mass() {
            static_assert(d == 1);
            return _mass; 
          }
          template<size_t d> std::vector<dCell_mass<_dimension,d>> mass() const {
            static_assert(d == 1);
            return _mass; 
          }
          template<size_t d> std::vector<dCell_traces<_dimension,d>> & traces() {
            static_assert(d == 1);
            return _traces; 
          }
          template<size_t d> std::vector<dCell_traces<_dimension,d>> traces() const {
            static_assert(d == 1);
            return _traces; 
          }
      };

      dCellVariableList<dimension,dimension> _dCellList;
  };
  ///@}
}
#endif

