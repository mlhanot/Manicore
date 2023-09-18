#ifndef MAP_INTERFACE_HPP_INCLUDED
#define MAP_INTERFACE_HPP_INCLUDED

#include "definitions.hpp"

/** \file map_interface.hpp
  Interface used to provide a mesh.

  \note
  The embedding reuses the structure of the cells mapping, 
  but only the mapping from the chart into the embedding (I and DI) is ever used.
  It is not necessary to define the reverse mapping (J and DJ).
  */

/// \addtogroup MeshFormat
///@{

extern "C"
{
  Manicore::ParametrizedMap<3,2> *List_embedding_2to3(size_t id);
  Manicore::ParametrizedDerivedMap<3,2> *List_pullback_2to3(size_t id);

  Manicore::ParametrizedMetricMap<2> *List_metrics_2D(size_t id); // detI is the volume form, and DJ the metric on the cotangent space

  Manicore::ParametrizedMap<2,1> *List_edge_maps_2D(size_t id);
  Manicore::ParametrizedDerivedMap<2,1> *List_edge_pullbacks_2D(size_t id);
  Manicore::ParametrizedMap<2,2> *List_face_maps_2D(size_t id);
  Manicore::ParametrizedDerivedMap<2,2> *List_face_pullbacks_2D(size_t id);
}
///@}

#endif

