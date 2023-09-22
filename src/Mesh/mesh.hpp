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

#ifndef MESH_HPP_INCLUDED
#define MESH_HPP_INCLUDED

#include "dcell.hpp"
#include "loader.hpp" // Needed to destruct the loader

#include <memory>

/** @defgroup Mesh
  @brief Classes to build and manage the mesh
  */

/** \file mesh.hpp
  Compute and store the topological and geometrical data of the mesh
  */
namespace Manicore {

  template<size_t> class Mesh_builder;

  /// \addtogroup Mesh
  ///@{

  /// Main data structure for the mesh
  /**
    The class is intended to be build with Mesh_builder, there is no public constructor.

    Store a collection of dCell_graph and dCell_map and keep the shared library describing the mesh.

    A dcell is a sub manifold of dimension d. It is defined by its parametrization into some reference charts.
    Some cells appears in several charts, hence, they have several parametrizations.
    The map_ids property of a cell give the id of the chart corresponding to each parametrizations (e.g. map_ids[0] is the chart corresponding to the first parametrization ...).

    \tparam dimension %Dimension of the manifold
    */
  template<size_t dimension>
    class Mesh 
    {
      public:
        /// Return the number of cell of a given dimension
        size_t n_cells(size_t d /*!< %Dimension of the cell */) const {return _cells_graph[d].size();}
        
        // Interface to the graph structure
        /// Return the map_ids of the i_cell-th d-cell
        std::vector<size_t> const & get_map_ids(size_t d /*!< %Dimension of the cell */,
                                                size_t i_cell /*!< Index of the cell */) const 
          {return _cells_graph[d][i_cell].get_map_ids();}

        /// Return a list filled with the global index of the elements on the boundary of the given dimension
        std::vector<size_t> const & get_boundary(size_t d_boundary /*!< %Dimension of the boundary cells */, 
                                                 size_t d /*!< %Dimension of the cell */,
                                                 size_t i_cell /*!< Index of the cell */) const 
          {return _cells_graph[d][i_cell].get_boundary(d_boundary);}

        /// The index for the mapping (of index 0) used on the element among the mappings of the boundary
        /** The first parametrization of an element correspond to a given chart U, 
          this function return the index of the parametrization of the element on its boundary corresponding to this chart U. */
        std::vector<size_t> const & get_relative_map(size_t d_boundary /*!< %Dimension of the boundary cell */, 
                                                     size_t d /*!< %Dimension of the cell */,
                                                     size_t i_cell /*!< Index of the cell */) const
          {return _cells_graph[d][i_cell].get_relative_map(d_boundary);}

        /// Return the local index of the i_b d_boundary-cell with respect to the icell-th d-cell
        /** Search among the boundary elements of the i_cell-th d-cell for a d_boundary-cell with global index i_b.
          Return Mesh::get_boundary (d_boundary,d,i_cell).size() on failure (and abort when debugging).
          */
        size_t global_to_local(size_t d_boundary /*!< %Dimension of the boundary cell */,
                               size_t d /*!< %Dimension of the cell */,
                               size_t i_b /*!< \e Global index of the boundary cell */,
                               size_t i_cell /*!< Index of the cell */) const 
          {return _cells_graph[d][i_cell].global_to_local(d_boundary,i_b);}

        /// Return the relative orientation of the j-th (d-1)-boundary of the i-th d-cell
        /** Multiply by this factor when doing an integration on the boundary */
        int get_boundary_orientation(size_t d /*!< %Dimension of the cell */, 
                                     size_t i_cell /*!< Index of the cell */, 
                                     size_t j_bd /*!< Relative index of the boundary cell */) const; 

        /// Interface to the extrinsic mapping
        /** Optional function to help viewing the embedding of the manifold. 
          This is not used in any computation and can be set freely in the mesh shared library */
        Eigen::Vector3d get_3D_embedding (size_t m_id /*!< Id of the embedding to use */,
                                          Eigen::Vector<double,dimension> const &x /*!< Location in the chart */) const 
        {return _maps[m_id]->I(x);}
        /// Interface to the extrinsic mapping
        /** Optional function to help viewing the embedding of the manifold. 
          This is not used in any computation and can be set freely in the mesh shared library 

          Using the differential of I should always map into the correct tangent space, 
          whereas J could introduce a deviation if it is not constant along the normal component.
         */
        Eigen::Matrix<double,3,dimension> get_3D_pushforward (size_t m_id /*!< Id of the embedding to use */,
                                          Eigen::Vector<double,dimension> const &x /*!< Location in the chart */) const 
        {return _maps_pullback[m_id]->DI(x);}

        // Access to the geometric structure
        /// Return the i-th dCell_map
        /** \tparam d %Dimension of the cell */
        template<size_t d> dCell_map<dimension,d> const & get_cell_map(size_t i) const;

        // Access to the metric structure
        /// Evaluate the metric (of the tangent space)
        Eigen::Matrix<double,dimension,dimension> metric (size_t m_id /*!< Id of the chart to use */,
                                                          Eigen::Vector<double,dimension> const &x /*!< Location in the chart */) const 
        {return _metric_maps[m_id]->metric(x);} // On the tangent space

        /// Evaluate the metric (of the cotangent space)
        Eigen::Matrix<double,dimension,dimension> metric_inv (size_t m_id /*!< Id of the chart to use */,
                                                              Eigen::Vector<double,dimension> const &x /*!< Location in the chart */) const 
        {return _metric_maps[m_id]->metric_inv(x);} // On the cotangent space

        /// Evaluate the scaling factor of the volume form
        double volume_form (size_t m_id /*!< Id of the chart to use */,
                            Eigen::Vector<double,dimension> const &x /*!< Location in the chart */) const 
        {return _metric_maps[m_id]->volume(x);}

        /// Return the relative orientation of a top dimensional cell with respect to the manifold
        int orientationTopCell(size_t i_cell) const
        {
          return _metric_maps[get_map_ids(dimension,i_cell)[0]]->orientation;
        }

        /// Evaluate the hodge star operator
        /** \tparam k Form degree
          \tparam d %Dimension of the cell
          \return Matrix from \f$ \Lambda^k(\mathbb{R}^d) \f$ to \f$ \Lambda^{d-k}(\mathbb{R}^d) \f$ 
         */
        template<size_t k,int d> Eigen::Matrix<double,Dimension::ExtDim(d-k,d),Dimension::ExtDim(k,d)>
          getHodge(size_t i_cell /*!< Index of the cell */, 
                   Eigen::Vector<double,d> const &x /*!< Location on the reference element of the cell*/) const; // Compute the Hodge star operator on the i_cell-th d-cell at x

      private:
        static constexpr size_t _max_dim = 3;
        Mesh() requires(dimension <= _max_dim) {;}
        friend class Mesh_builder<dimension>;
        typename dCell_graph<dimension>::Collection _cells_graph;
        std::unique_ptr<Maps_loader<dimension>> _loader_ref; // Tie the life of the shared library to the mesh (must be destroyed AFTER every pointer to any mapping 
        std::vector<std::unique_ptr<ParametrizedMap<3,dimension>>> _maps;
        std::vector<std::unique_ptr<ParametrizedDerivedMap<3,dimension>>> _maps_pullback;
        std::vector<std::unique_ptr<ParametrizedMetricMap<dimension>>> _metric_maps;
        // TODO use the generic solution of the class PEC
        std::vector<dCell_map<dimension,0>> _geo0;
        std::vector<dCell_map<dimension,1>> _geo1;
        std::vector<dCell_map<dimension,2>> _geo2;
        std::vector<dCell_map<dimension,3>> _geo3;
    };
  ///@}

  // ---------------------------------------------------------------------------------------------------------
  // Implementation
  // ---------------------------------------------------------------------------------------------------------
    template<size_t dimension>
    int Mesh<dimension>::get_boundary_orientation(size_t d, size_t i_cell, size_t j_bd) const
    {
      size_t i_bd_abs = get_boundary(d-1,d,i_cell)[j_bd];
      size_t bd_rel_map = get_relative_map(d-1,d,i_cell)[j_bd];
      auto get_orientation = [&]<size_t _d>(auto&& get_orientation)
      {
        if constexpr(_d == dimension+1) {
          assert(false && "Dimension too high or low");
          return 0;
        } else {
          if (_d == d) {
            auto const & E = get_cell_map<_d-1>(i_bd_abs);
            auto x = Geometry::middleSimplex<_d-1>(E.get_reference_elem()[0]);
            auto pM = E.evaluate_DI(bd_rel_map,x);
    if (d == dimension) {
            return get_cell_map<_d>(i_cell).get_orientation(0,E.evaluate_I(bd_rel_map,x),pM)*orientationTopCell(i_cell);
    } else {
            return get_cell_map<_d>(i_cell).get_orientation(0,E.evaluate_I(bd_rel_map,x),pM);
    }
          } else {
            return get_orientation.template operator()<_d+1>(get_orientation);
          }
        }
      };
      return get_orientation.template operator()<2>(get_orientation); // Start at dim(T) = 2, since the boundary must be at least of dimension 1
    }

    // Doxygen wrongly assumes this is an overload, prevent it from duplicating the entry
    /// @cond
    template<size_t dimension>
    template<size_t d> 
    dCell_map<dimension,d> const & Mesh<dimension>::get_cell_map(size_t i) const
    {
      static_assert(d <= _max_dim);
      if constexpr (d == 0) return _geo0[i];
      else if constexpr (d == 1) return _geo1[i];
      else if constexpr (d == 2) return _geo2[i];
      else return _geo3[i];
    }
    /// @endcond

    template<size_t dimension>
    template<size_t k,int d> Eigen::Matrix<double,Dimension::ExtDim(d-k,d),Dimension::ExtDim(k,d)>
    Mesh<dimension>::getHodge(size_t i_cell, Eigen::Vector<double,d> const &x) const 
    {
      auto const & F = get_cell_map<d>(i_cell);
      auto const Ix = F.evaluate_I(0,x);
      Eigen::Matrix<double,d,d> invGF;
      double sqrtdetG;
      if constexpr(d == dimension) { // TODO check if this is actually faster
        auto const DJ = F.evaluate_DJ(0,Ix);
        invGF = DJ*metric_inv(get_map_ids(d,i_cell)[0],Ix)*DJ.transpose();
        sqrtdetG = volume_form(get_map_ids(d,i_cell)[0],Ix) * std::abs(F.evaluate_DI(0,x).determinant())*orientationTopCell(i_cell);
      } else {
        auto const DI = F.evaluate_DI(0,x);
        Eigen::Matrix<double,d,d> pullbackMetric = DI.transpose()*metric(get_map_ids(d,i_cell)[0],Ix)*DI;
        invGF = (pullbackMetric).inverse();
        sqrtdetG = std::sqrt(pullbackMetric.determinant());
      }
      auto const partialDet = Compute_pullback<k,d,d>::compute(invGF);
      auto const &complBasis = ComplBasis<k,d>::compute();
      return sqrtdetG*complBasis*partialDet;
      // ComplBasis does not duplicate indices, hence there is no 1/(d-k)! term
    }

} // Namespace

#endif

