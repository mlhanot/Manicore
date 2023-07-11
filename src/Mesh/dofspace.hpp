#ifndef DOFSPACE_HPP_INCLUDED
#define DOFSPACE_HPP_INCLUDED

#include "mesh.hpp"

/** \file dofspace.hpp
  Provides the Local-Global mapping of unknowns
  */

namespace Manicore {
  /// \addtogroup Mesh
  ///@{

  /// Convert between local and global data
  /** \tparam dimension %Dimension of the manifold 
    \warning This does not take ownership of the mesh but keep a pointer of it. Ensure that the mesh survives this class.
    */
  template<size_t dimension>
  class DOFSpace {
    public:
      /// Empty space
      DOFSpace() {;}
      /** \warning Ensure that the mesh survives this class */
      DOFSpace(Mesh<dimension> const * mesh /*!< Mesh to use */,std::array<size_t,dimension+1> nb_local_dof /*!< Number of degree of freedom on cell of each dimension */) 
        : _mesh(mesh), _nb_local_dofs(nb_local_dof) {;}

      /// Return the number of degree of free on a d-cell
      size_t numLocalDofs(size_t d /*!< %Dimension of the cell */) const {return _nb_local_dofs[d];}
      /// Return the total number of Degree Of Freedoms
      size_t dimensionMesh() const;
      /// Return the number of Degree Of Freedoms of the unknown attached the i-th cell of dimension d (including its boundary).
      size_t dimensionCell(size_t d /*!< %Dimension of the cell */,
                           size_t i_cell /*!< Index of the cell */) const;
      /// Return the local offset of the unknown attached the i-th cell of dimension d.
      /** Return the index of the first unknown attached to the cell itself, or equivalently, the number of unknown attached to the boundary of the cell */
      size_t localOffset(size_t d /*!< %Dimension of the cell */,
                         size_t i_cell /*!< Index of the cell */) const;
      /// Return the local offset of the unknown attached the i_bd_rel-th cell of dimension d_boundary with respect to the i_cell-th cell of dimension d
      /** Example: localOffset (1,3,0,231) return the offset of the first edge (1-cell) of the 3-cell numbered 231 in the mesh */
      size_t localOffset(size_t d_boundary /*!< %Dimension of the cell on the boundary */, 
                         size_t d /*!< %Dimension of the cell */, 
                         size_t i_bd_rel /*!< Relative index of the boundary cell */, 
                         size_t i_cell /*!< Index of the cell */) const;
      /// Return the global offset of the unknown attached the i-th cell of dimension d.
      size_t globalOffset(size_t d /*!< %Dimension of the cell */,
                          size_t i_cell /*!< Index of the cell */) const;

      /// Restrict the vector vh to the i-th cell of dimension d (including its boundary).
      /** \return VectorXd of size DOFSpace::dimensionCell (d,i_cell) */
      Eigen::VectorXd restrict(size_t d /*!< %Dimension of the cell */, 
                               size_t i_cell /*!< Index of the cell */, 
                               const Eigen::VectorXd & vh /*!< Vector of global degree of freedom of size DOFSpace::dimensionMesh */) const;
      /// Extend operator op from a cell on the boundary to the i_cell-th cell.
      /** \return Matrix of size [ DOFSpace::dimensionCell (d,i_cell), n] */
      Eigen::MatrixXd extendOperator(size_t d_boundary /*!< %Dimension of the boundary cell */, 
                                     size_t d /*!< %Dimension of the cell */, 
                                     size_t i_bd_global /*!< \e Global index of the boundary cell */, 
                                     size_t i_cell /*!< Index of the cell */, 
                                     const Eigen::MatrixXd & op /*!< Matrix of size [ DOFSpace::dimensionCell (d_boundary,i_bd_global), n] */) const;

    private:
      const Mesh<dimension>* _mesh;
      std::array<size_t,dimension+1> _nb_local_dofs;
  };
  ///@}

  // ---------------------------------------------------------------------------------------------------------
  // Implementation
  // ---------------------------------------------------------------------------------------------------------
  template<size_t dimension>
  size_t DOFSpace<dimension>::dimensionMesh() const 
  {
    size_t rv = _mesh->n_cells(0)*_nb_local_dofs[0];
    for (size_t i = 1; i <= dimension; ++i) {
      rv += _mesh->n_cells(i)*_nb_local_dofs[i];
    }
    return rv;
  }

  template<size_t dimension>
  size_t DOFSpace<dimension>::dimensionCell(size_t d, size_t i_cell) const 
  {
    size_t rv = _nb_local_dofs[d];
    for (size_t i = 0; i < d; ++i) {
      rv += _mesh->get_boundary(i,d,i_cell).size()*_nb_local_dofs[i];
    }
    return rv;
  }

  template<size_t dimension>
  size_t DOFSpace<dimension>::localOffset(size_t d, size_t i_cell) const 
  {
    size_t rv = 0;
    for (size_t i = 0; i < d; ++i) {
      rv += _mesh->get_boundary(i,d,i_cell).size()*_nb_local_dofs[i];
    }
    return rv;
  }

  template<size_t dimension>
  size_t DOFSpace<dimension>::localOffset(size_t d_boundary, size_t d, size_t i_bd_rel, size_t i_cell) const 
  {
    assert(d_boundary < d);
    size_t rv = i_bd_rel*_nb_local_dofs[d_boundary];
    for (size_t i = 0; i < d_boundary; ++i) {
      rv += _mesh->get_boundary(i,d,i_cell).size()*_nb_local_dofs[i];
    }
    return rv;
  }

  template<size_t dimension>
  size_t DOFSpace<dimension>::globalOffset(size_t d, size_t i_cell) const 
  {
    size_t rv = i_cell*_nb_local_dofs[d];
    for (size_t i = 0; i < d; ++i) {
      rv += _mesh->n_cells(i)*_nb_local_dofs[i];
    }
    return rv;
  }

} // end namespace

#endif


