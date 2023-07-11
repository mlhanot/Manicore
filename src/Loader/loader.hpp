#ifndef LOADER_HPP_INCLUDED
#define LOADER_HPP_INCLUDED

#include "map_interface.hpp"

/** @defgroup MeshFormat
  @brief Interface for the mesh files
  */

namespace Manicore {

  /** \file loader.hpp 
    Load the mesh description given as a shared library for use in Manicore */

  /// \addtogroup MeshFormat
  ///@{
  
  /// Class to load a shared library
  /** \tparam dimension %Dimension of the manifold
    */
  template<size_t dimension>
  class Maps_loader {
    public:
      Maps_loader(const char * /*!< Filename of the shared library */);
      ~Maps_loader();

      ParametrizedMap<3,dimension>* get_new_embedding_3D(size_t id) const;
      ParametrizedMap<3,dimension>* get_new_embedding_3D(size_t id, std::vector<double> const &extra) const;
      ParametrizedMetricMap<dimension>* get_new_metrics(size_t id) const;
      ParametrizedMetricMap<dimension>* get_new_metrics(size_t id, std::vector<double> const &extra) const;
      ParametrizedMap<dimension,1>* get_new_edge_map(size_t id) const;
      ParametrizedMap<dimension,1>* get_new_edge_map(size_t id, std::vector<double> const &extra) const;
      ParametrizedMap<dimension,2>* get_new_face_map(size_t id) const;
      ParametrizedMap<dimension,2>* get_new_face_map(size_t id, std::vector<double> const &extra) const;
      ParametrizedMap<dimension,3>* get_new_cell_map(size_t id) const;
      ParametrizedMap<dimension,3>* get_new_cell_map(size_t id, std::vector<double> const &extra) const;
      ParametrizedDerivedMap<dimension,1>* get_new_edge_pullbacks(size_t id) const;
      ParametrizedDerivedMap<dimension,1>* get_new_edge_pullbacks(size_t id, std::vector<double> const &extra) const;
      ParametrizedDerivedMap<dimension,2>* get_new_face_pullbacks(size_t id) const;
      ParametrizedDerivedMap<dimension,2>* get_new_face_pullbacks(size_t id, std::vector<double> const &extra) const;
      ParametrizedDerivedMap<dimension,3>* get_new_cell_pullbacks(size_t id) const;
      ParametrizedDerivedMap<dimension,3>* get_new_cell_pullbacks(size_t id, std::vector<double> const &extra) const;

    private:
      void *_handle;
      ParametrizedMap<3,dimension>* (*_3D_embedding)(size_t);
      ParametrizedMetricMap<dimension>* (*_metrics)(size_t);
      ParametrizedMap<dimension,1>* (*_edge_maps)(size_t);
      ParametrizedMap<dimension,2>* (*_face_maps)(size_t);
      ParametrizedMap<dimension,3>* (*_cell_maps)(size_t);
      ParametrizedDerivedMap<dimension,1>* (*_edge_pullbacks)(size_t);
      ParametrizedDerivedMap<dimension,2>* (*_face_pullbacks)(size_t);
      ParametrizedDerivedMap<dimension,3>* (*_cell_pullbacks)(size_t);
      void _setup_maps();
  };
  ///@}

} // end namespace

#endif

