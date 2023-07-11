#ifndef MESH_BUILDER_HPP_INCLUDED
#define MESH_BUILDER_HPP_INCLUDED

#include "mesh.hpp"

/** \file mesh_builder.hpp
  Build the mesh from a json and a shared library
  */

namespace Manicore {
  /// Build the internal representation of the mesh
  /** \tparam dimension %Dimension of the manifold */
  template<size_t dimension> 
    class Mesh_builder {
      public:
      /// \addtogroup Mesh
      ///@{

        /// Construct a Mesh object from files
        static Mesh<dimension> * build(const char * meshfile /*!< json file describing the topology, the format is given in <a href="mesh_format" target = "_blank"><b>mesh_format</b></a> */,
                                       const char *mapfile /*!< shared library describing the geometry, the format is given in map_interface.hpp */);
        ///@}
    };
} // end namespace

#endif 

