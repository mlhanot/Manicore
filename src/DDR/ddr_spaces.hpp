#ifndef DDR_SPACES_HPP
#define DDR_SPACES_HPP

#include "dofspace.hpp"
#include "pec.hpp"

#include <memory>
#include <iostream>

/** \file ddr_spaces.hpp
  Discrete operators for DDR-PEC
  */
namespace Manicore {

  /// \addtogroup DDR
  ///@{

  /// Implement the discrete operators of DDR-PEC
  /** Interpolator, discrete differential and potential reconstruction.

    This class compute store a matrix for each operator on each cell.
    It also create and store the corresponding PEC object (holding the masses and traces of the cells)

    \tparam dimension %Dimension of the manifold
    \warning This does not take ownership of the mesh but keep a pointer of it. Ensure that the mesh survives this class.
    */
  template<size_t dimension>
  class DDR_Spaces {
    public:
      /** \warning Ensure that the mesh survives this class */
      DDR_Spaces(Mesh<dimension> const & mesh, /*!< Mesh to use */
                 int r, /*!< Polynomial degree */
                 bool use_threads = true, /*!< Enable pthreads parallelism */
                 std::array<int,dimension> const * dqr_p = nullptr, /*!< Degree of quadrature used to generate the mass matrices. It cannot be exact for generic metric and default to a pretty high degree. Use a lower degree if the metric and cells are almost flat. */
                 std::ostream & output = std::cout /*!< Output stream for status messages. */
                 );

      /// Signature of the function to use in interpolate
      /** \tparam k Form degree */
      template<size_t k>
      using FunctionType = std::function<Eigen::Vector<double,Dimension::ExtDim(k,dimension)>(size_t map_id, const Eigen::Vector<double,dimension> &)>;

      /// Interpolate the given function on the discrete spaces
      /** \tparam k Form degree */
      template<size_t k >
      Eigen::VectorXd interpolate(FunctionType<k> const & func, /*!< Function to interpolate */
                                  std::array<int,dimension> const * dqr_p = nullptr /*!< Quadrature degree to use */) const;

      /// Compute the full differential operator
      /** \return \f$\star d_r^k \f$ in \f$\mathcal{P}_r\Lambda^{d-k-1}(\mathbb{R}^d)\f$ */
      const Eigen::MatrixXd & full_diff(size_t k /*!<Form degree*/, size_t d /*!< Cell dimension */,size_t i_cell /*!< Cell index */) const // Return \star d in PL(r,d-k-1,d)
      {
        assert(d <= dimension && k < d && i_cell < _ops[d].size() && "Access of diff out of range");
        return _ops[d][i_cell].full_diff[k];
      }
      /// Compute the projected differential operator
      /** \return \f$\star \underline{d}_r^k\f$ on a cell including its boundary in \f$\cup_{d' = k}^{d} \mathcal{P}_r\Lambda^{d-k-1}(\mathbb{R}^{d'})\f$ */
      Eigen::MatrixXd compose_diff(size_t k /*!<Form degree*/, size_t d /*!< Cell dimension */,size_t i_cell /*!< Cell index */) const; 
      /// Compute the potential operator
      /** \return \f$\star P_r^k\f$ in \f$\mathcal{P}_r\Lambda^{d-k}(\mathbb{R}^d)\f$ */
      const Eigen::MatrixXd & potential(size_t k /*!<Form degree*/, size_t d /*!< Cell dimension */,size_t i_cell /*!< Cell index */) const // Return \star P^k in PL(r,d-k,d)
      {
        assert(d <= dimension && k <= d && i_cell< _ops[d].size() && "Access of potential out of range");
        return _ops[d][i_cell].P[k];
      }

      /// Return DOFSpace associated to k forms
      DOFSpace<dimension> const & dofspace(size_t k/*!< Form degree*/) const {
        return _dofspace[k];
      }
      /// Return the Mesh associated with the class
      const Mesh<dimension>* mesh() const {return _mesh;}
      /// Return the polynomial degree associated with the class
      int degree() const {return _r;}

      /// Compute the local \f$L^2\f$ product (including the cell boundary)
      /** The contribution on the cell is \f$\int_f \langle P^k , P^k \rangle \text{vol}_f \f$.
        The contribution from a cell \f$f' \in \partial f \f$ of the boundary is \f$\int_f \langle \text{tr}_{f'} P^k_f - P^k_{f'} , \text{tr}_{f'} P^k_f - P^k_{f'} \rangle \text{vol}_{f'} \f$ multiplied by a scaling factor.
        */
      Eigen::MatrixXd computeL2Product(size_t k /*!<Form degree*/,size_t i_cell /*!< Cell index */) const;

    private:
      struct DDR_Operators { // one for each form degree k
        std::array<Eigen::MatrixXd,dimension+1> full_diff; // equal to \star d, PL(r,d-k-1,d) valued
        std::array<Eigen::MatrixXd,dimension+1> diff; // equal to \star \ul{d}, PLtrimmed(r,d-k-1,d) valued
        std::array<Eigen::MatrixXd,dimension+1> P; // equal to \star P, PL(r,d-k,d) valued
      };

      const Mesh<dimension>* _mesh;
      int _r;
      bool _use_threads;
      std::unique_ptr<PEC<dimension>> _ddr, _ddr_po;
      std::array<DOFSpace<dimension>,dimension+1> _dofspace; // one for all form degree
      std::array<std::vector<DDR_Operators>,dimension+1> _ops; // one for each cell dimension
  };
  ///@}
}
#endif

