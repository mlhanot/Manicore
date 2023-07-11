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

#ifndef DCELL_HPP_INCLUDED
#define DCELL_HPP_INCLUDED

#include "exterior_algebra.hpp"
#include "geometry.hpp"

#include <vector>
#include <set>
#include <memory>

/** \file dcell.hpp
  Store the geometrical description of each cell and their topological relations
  */

namespace Manicore {
  /// \addtogroup Mesh
  ///@{

  /// Manage topological relations between cells
  /** \tparam dimension %Dimension of the manifold 
    This class should not be used directly but through Mesh instead.
   */
  template<size_t dimension>
  class dCell_graph {
    public:
      /// Type to store the topological data of every cell
      typedef std::array<std::vector<dCell_graph>,dimension+1> Collection;

      /// Constructor for vertex (0-cell)
      dCell_graph(size_t id /*!< Index of the cell being constructed */, 
                  std::vector<size_t> const & map_ids /*!< Id of the charts corresponding to its parametrization */);
      /// General constructor
      /** Cannot be used for vertex */
      dCell_graph(Collection const & /*!< Collection of all lower dimension cells */,
                  size_t d /*!< Dimension of the cell being constructed */, 
                  size_t id /*!< Index of the cell being constructed */, 
                  std::vector<size_t> const & map_ids /*!< Id of the charts corresponding to its parametrization */, 
                  std::vector<size_t> const & bd /*!< List of the global index of the (d-1)-cells on its boundary */);

      /// Return the global index of the cell
      size_t get_id() const {return _bd_ids[_d][0];}
      /// Return the id of the charts corresponding to its parametrizations
      std::vector<size_t> const & get_map_ids() const {return _map_ids;}
      /// Return the global index of the cell of dimension dim on its boundary
      std::vector<size_t> const & get_boundary(size_t dim /*!< %Dimension of the boundary cell */) const {return _bd_ids[dim];}
      /// Return the index of the parametrization of the cell of dimension dim on the boundary, corresponding to same chart as the first parametrization of this cell
      std::vector<size_t> const & get_relative_map(size_t dim /*!< %Dimension of the boundary cell */) const {return _bd_rel_maps[dim];}
      /// Return the index of the i_b d_boundary-cell relative to this cell
      size_t global_to_local(size_t d_boundary /*!< %Dimension of the boundary cell */,size_t i_b /*!< Global index of the boundary cell */) const;
    private:
      size_t _d;
      std::vector<size_t> _map_ids;
      std::array<std::vector<size_t>,dimension+1> _bd_ids; 
      std::array<std::vector<size_t>,dimension> _bd_rel_maps; // relative ids of the corresponding map in the boundary
      size_t _find_map(size_t id) const;
  };
  
  /// Manage the geometry of a cell
  /** \tparam dimension %Dimension of the manifold 
    \tparam d %Dimension of the cell 
    Hold the reference element, compute the mappings and pullback between the reference element and the charts.

    The class is specialize to apply some optimization when dealing with flat cells.
    */
  template<size_t dimension,size_t d>
  class dCell_map {
    public:
      /// %Dimension of the cell
      static constexpr size_t cell_dim = d;

      /// Flat constructor
      /** Simplify the structure when dealing with flat cells.
        This must be a flat surface in a chart parametrized by an affine mapping, and all its boundary element must also be flat.
        As an additional restriction, a flat cell can only be in a single chart 
       */
      dCell_map(Eigen::Vector<double,dimension> const &center_mass /*!< Centroid */,
                Eigen::Matrix<double,dimension,d> const &flat_map /*!< Basis of the tangent space */,
                double diam /*!< Scaling factor to apply */,
                std::vector<Simplex<d>> const &triangulation /*!< Reference element given as a collection of simplexes */);
      /// General constructor
      dCell_map(std::vector<std::unique_ptr<ParametrizedMap<dimension,d>>> & maps, /*!< List of the parametrization of this cell */
                std::vector<std::unique_ptr<ParametrizedDerivedMap<dimension,d>>> & pullback_maps /*!< Differentials of the parametrization of this cell */,
                std::vector<Simplex<d>> const &triangulation /*!< Reference element given as a collection of simplexes */);

      /// Is the cell flat
      /** Only check that the cell was constructed with the flat constructor, does not check if the parametrization is flat 
        */
      bool is_flat() const {return _is_flat;}
      // Geometry
      /// Return the reference element as a collection of simplexes
      std::vector<Simplex<d>> const & get_reference_elem() const {return _triangulation;}

      /// Evaluate the parametrization from the reference element to a chart
      Eigen::Vector<double,dimension> evaluate_I(size_t rel_map_id /*!< Relative id of the chart to use */, 
                                                 Eigen::Vector<double,d> const &x /*!< Location on the reference element */) const;

      /// Evaluate the inverse mapping from the chart to the reference element
      Eigen::Vector<double,d> evaluate_J(size_t rel_map_id /*!< Relative id of the chart to use */, 
                                         Eigen::Vector<double,dimension> const &x /*!< Location on the chart*/) const;
      // Metric
      /// Evaluate a scalar polynomial on the reference element
      double evaluate_poly_on_ref(Eigen::Vector<double,d> const &x /*!< Location on the reference element */,
                                  size_t i_pbasis /*!< Index of the polynomial to evaluate */, 
                                  int r /*!< Polynomial basis */) const;

      /// Evaluate the pullback of a scalar polynomial on the chart
      double evaluate_poly_pullback(size_t rel_map_id /*!< Relative id of the chart to use */, 
                                    Eigen::Vector<double,dimension> const &x /*!< Location on the chart*/,
                                    size_t i_pbasis /*!< Index of the polynomial to evaluate */,
                                    int r /*!< Polynomial basis */) const;

      /// Evaluate the differential of the parametrization
      Eigen::Matrix<double,dimension,d> evaluate_DI(size_t rel_map_id /*!< Relative id of the chart to use */, 
                                                    Eigen::Vector<double,d> const &x /*!< Location on the reference element */) const;

      /// Evaluate the differential of the inverse mapping
      Eigen::Matrix<double,d,dimension> evaluate_DJ(size_t rel_map_id /*!< Relative id of the chart to use */,
                                                    Eigen::Vector<double,dimension> const &x /*!< Location on the chart*/) const;

      /// Compute the action of the pullback of l-forms by the parametrization
      /** \tparam l Form degree */
      template<size_t l> Eigen::Matrix<double,Dimension::ExtDim(l,d),Dimension::ExtDim(l,dimension)> 
        evaluate_DI_p(size_t rel_map_id /*!< Relative id of the chart to use */, 
                      Eigen::Vector<double,d> const &x /*!< Location on the reference element */) const; // Return the pullback by I of the basis of l-forms

      /// Compute the action of the pullback of l-forms by the inverse mapping
      /** \tparam l Form degree */
      template<size_t l> Eigen::Matrix<double,Dimension::ExtDim(l,dimension),Dimension::ExtDim(l,d)> 
        evaluate_DJ_p(size_t rel_map_id /*!< Relative id of the chart to use */, 
                      Eigen::Vector<double,dimension> const &x /*!< Location on the chart*/) const; // Return the pullback by J of the basis of l-forms

      // Orientation
      /// Return the relative orientation
      /** Given a point on the boundary x, it attempt to find which simplex of the reference element S contains x,
        then compute a outward normal vector subtracting the center of S to x.
        The outward normal is then compared with a basis of the tangent space of the boundary to get the orientation of the boundary.
        \warning This can fail in some corner case, for example when taking x to be a vertices, or if the internal tolerance differs too much from the scaling of the element.
        */
      int get_orientation(size_t rel_map_id /*!< Relative id of the chart to use */, 
                             Eigen::Vector<double,dimension> const &x /*!< Any point on the boundary in the chart (avoid vertices) */,
                             Eigen::Matrix<double,dimension,d-1> const &pM /*!< Matrix of a vector basis of the tangent space of the boundary*/
                             ) const requires(d>1);

    private:
      Eigen::Vector<double,dimension> _center_mass; // Used when _is_flat is true 
      Eigen::Matrix<double,dimension,d> _flat_map; // Used when _is_flat is true and d<dimension
      double _diam; // Used when _is_flat is true
      std::vector<std::unique_ptr<ParametrizedMap<dimension,d>>> _maps; // Used when _is_flat is false
      std::vector<std::unique_ptr<ParametrizedDerivedMap<dimension,d>>> _pullback_maps; // Used when _is_flat is false
      const bool _is_flat; // Element and all its boundaries are flat, it must also correspond to a single map for d>0

      std::vector<Simplex<d>> _triangulation; // Used to compute integrals
  };

  /// Specialization for the vertices
  /** \tparam dimension %Dimension of the manifold */
  template<size_t dimension>
  class dCell_map<dimension,0> { // Specialize for vertices
    public:
      dCell_map(std::vector<Eigen::Vector<double,dimension>> const &coords /*!< Locations of the vertex within each chart */) : _locs(coords) {};

      /// Return the location of the vertex in a given chart
      Eigen::Vector<double,dimension> const & coord(size_t map_rel_id /*!< Relative id of the chart to use */) const {return _locs[map_rel_id];}

    private:
      std::vector<Eigen::Vector<double,dimension>> _locs;
  };
  ///@}

  // ---------------------------------------------------------------------------------------------------------
  // Implementation
  // ---------------------------------------------------------------------------------------------------------
  // ---------------------------------------------------------------------------------------------------------
  // Graph

  template<size_t dimension>
  dCell_graph<dimension>::dCell_graph(size_t id, std::vector<size_t> const & map_ids)
  : _d(0), _map_ids(map_ids) {
    _bd_ids[_d] = std::vector<size_t>{id};
  }

  template<size_t dimension>
  dCell_graph<dimension>::dCell_graph(Collection const &coll,size_t d, size_t id, std::vector<size_t> const & map_ids, std::vector<size_t> const & bd) 
  : _d(d), _map_ids(map_ids) {
    assert(d > 0);
    // Insert map of faces
    _bd_ids[d] = std::vector<size_t>{id};
    _bd_ids[d-1] = bd;
    for (size_t i = 0; i < bd.size(); ++i) {
      _bd_rel_maps[d-1].push_back(coll[d-1][bd[i]]._find_map(_map_ids[0]));
    }
    // Iterate lower dimensions
    for (size_t dim = d-1; dim > 0; --dim) {
      std::set<size_t> clist;
      for (size_t i = 0; i < _bd_ids[dim].size(); ++i) {
        std::vector<size_t> const & cb = coll[dim][_bd_ids[dim][i]].get_boundary(dim-1);
        for (size_t j = 0; j < cb.size(); ++j) {
          clist.insert(cb[j]);
        }
      }
      for (auto const &p : clist) {
        _bd_ids[dim-1].push_back(p);
        _bd_rel_maps[dim-1].push_back(coll[dim-1][p]._find_map(_map_ids[0]));
      }
    }
  }
  
      // This is relative to the principal map of the object (to _map_ids[0])
  template<size_t dimension>
  size_t dCell_graph<dimension>::_find_map(size_t id) const {
    for (size_t i = 0; i < _map_ids.size(); ++i) {
      if (_map_ids[i] == id) return i;
    }
    return _map_ids.size();
  }

  template<size_t dimension>
  size_t dCell_graph<dimension>::global_to_local(size_t d_boundary,size_t i_b) const
  {
    for (size_t i = 0; i < _bd_ids[d_boundary].size(); ++ i) {
      if (_bd_ids[d_boundary][i] == i_b) return i;
    }
    assert(false);
    return _bd_ids[d_boundary].size();
  }

  // ---------------------------------------------------------------------------------------------------------
  // Map

  template<size_t dimension,size_t d>
  dCell_map<dimension,d>::dCell_map(Eigen::Vector<double,dimension> const &center_mass,
                                    Eigen::Matrix<double,dimension,d> const &flat_map,
                                    double diam, std::vector<Simplex<d>> const &triangulation)
    : _center_mass(center_mass), _flat_map(flat_map), _diam(diam), _is_flat(true), _triangulation(triangulation) 
  {}

  template<size_t dimension,size_t d>
  dCell_map<dimension,d>::dCell_map(std::vector<std::unique_ptr<ParametrizedMap<dimension,d>>> &maps,
                std::vector<std::unique_ptr<ParametrizedDerivedMap<dimension,d>>> & pullback_maps,
                std::vector<Simplex<d>> const &triangulation)
  : _maps(std::move(maps)), _pullback_maps(std::move(pullback_maps)), _is_flat(false), _triangulation(triangulation)
  {}

  template<size_t dimension,size_t d>
  Eigen::Vector<double,dimension> dCell_map<dimension,d>::evaluate_I(size_t rel_map_id, Eigen::Vector<double,d> const &x) const
  {
    if (_is_flat) {
      if constexpr(dimension == d) {
        return x*_diam  + _center_mass;
      } else {
        return _flat_map*x*_diam + _center_mass;
      }
    } else {
      assert(rel_map_id < _maps.size());
      return _maps[rel_map_id]->I(x);
    }
  }

  template<size_t dimension,size_t d>
  Eigen::Vector<double,d> dCell_map<dimension,d>::evaluate_J(size_t rel_map_id, Eigen::Vector<double,dimension> const &x) const
  {
    if (_is_flat) {
      if constexpr(dimension == d) {
        return (x - _center_mass)/_diam;
      } else {
        return _flat_map.transpose()*(x - _center_mass)/_diam;
      }
    } else {
      assert(rel_map_id < _maps.size());
      return _maps[rel_map_id]->J(x);
    }
  }

  /// -----------------------------------------------------------------------------------------------------------
  // Evaluations
  template<size_t dimension,size_t d>
  double dCell_map<dimension,d>::evaluate_poly_on_ref(Eigen::Vector<double,d> const &x,size_t i_pbasis,int r) const 
  {
    assert((i_pbasis < Dimension::PolyDim(r,d)) && "Index out of range");
    auto const & power = Monomial_powers<d>::complete(r)[i_pbasis];
    double rv = std::pow(x(0),power[0]);
    for (size_t i = 1; i < d; ++i) {
      rv *= std::pow(x(i),power[i]);
    }
    return rv;
  }
  
  template<size_t dimension,size_t d>
  double dCell_map<dimension,d>::evaluate_poly_pullback(size_t rel_map_id, Eigen::Vector<double,dimension> const &x,size_t i_pbasis,int r) const 
  {
    return evaluate_poly_on_ref(evaluate_J(rel_map_id,x),i_pbasis,r);
  }

  template<size_t dimension,size_t d>
  Eigen::Matrix<double,dimension,d> dCell_map<dimension,d>::evaluate_DI(size_t rel_map_id, Eigen::Vector<double,d> const &x) const
  {
    if (_is_flat) {
      if constexpr(dimension == d) {
        return Eigen::Matrix<double,d,d>::Identity()*_diam;
      } else {
        return _flat_map*_diam;
      }
    } else {
      return _pullback_maps[rel_map_id]->DI(x);
    }
  }

  template<size_t dimension,size_t d>
  Eigen::Matrix<double,d,dimension> dCell_map<dimension,d>::evaluate_DJ(size_t rel_map_id, Eigen::Vector<double,dimension> const &x) const
  {
    if (_is_flat) {
      if constexpr(dimension == d) {
        return Eigen::Matrix<double,d,d>::Identity()/_diam;
      } else {
        return _flat_map/_diam;
      }
    } else {
      return _pullback_maps[rel_map_id]->DJ(x);
    }
  }

  template<size_t dimension,size_t d>
    template<size_t l>
    Eigen::Matrix<double,Dimension::ExtDim(l,d),Dimension::ExtDim(l,dimension)> dCell_map<dimension,d>::evaluate_DI_p(size_t rel_map_id, Eigen::Vector<double,d> const &x) const
  {
    if (_is_flat) {
      if constexpr(dimension == d) {
        return Eigen::Matrix<double,Dimension::ExtDim(l,d),Dimension::ExtDim(l,d)>::Identity()*std::pow(_diam,l);
      } else {
        return Compute_pullback<l,d,dimension>::compute(_flat_map*_diam);
      }
    } else {
      return Compute_pullback<l,d,dimension>::compute(_pullback_maps[rel_map_id]->DI(x));
    }
  }

  template<size_t dimension,size_t d>
    template<size_t l>
    Eigen::Matrix<double,Dimension::ExtDim(l,dimension),Dimension::ExtDim(l,d)> dCell_map<dimension,d>::evaluate_DJ_p(size_t rel_map_id, Eigen::Vector<double,dimension> const &x) const
  {
    if (_is_flat) {
      if constexpr(dimension == d) {
        return Eigen::Matrix<double,Dimension::ExtDim(l,d),Dimension::ExtDim(l,d)>::Identity()*std::pow(1/_diam,l);
      } else {
        return Compute_pullback<l,dimension,d>::compute(_flat_map.transpose()/_diam);
      }
    } else {
      return Compute_pullback<l,dimension,d>::compute(_pullback_maps[rel_map_id]->DJ(x));
    }
  }

  template<size_t dimension,size_t d>
  int dCell_map<dimension,d>::get_orientation(size_t rel_map_id, 
      Eigen::Vector<double,dimension> const &x,
      Eigen::Matrix<double,dimension,d-1> const &pM) const requires(d>1) 
  {
    auto const Jx = evaluate_J(rel_map_id,x);
    for (auto const &S : _triangulation) {
      if (Geometry::inside<d>(Jx,S)) {
        auto mid = Geometry::middleSimplex<d>(S);
        auto DI = evaluate_DI(rel_map_id,Jx);
        Eigen::Matrix<double,dimension,d> cmpM;
        cmpM.rightCols(d-1) = pM;
        cmpM.leftCols(1) = DI*(Jx - mid).normalized();
        double tmp = (evaluate_DJ(rel_map_id,x)*cmpM).determinant(); 
        assert(std::abs(tmp) > 1e-10 && "Volume form is 0");
        return (tmp > 0.)? 1 : -1;
      }
    }
    assert(false && "Point outside of element");
    return 0.;
  }

} // Namespace

#endif

