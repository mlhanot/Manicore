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

#include "mesh_builder.hpp"

#include "geometry.hpp"

#include <fstream>
#include <json.hpp>

using json = nlohmann::json;

using namespace Manicore;

template<size_t dimension, size_t d> bool is_boundary_flat(Mesh<dimension> const *mesh,std::vector<size_t> const & bd_list) {
  bool flat = true;
  for (size_t i = 0; i < bd_list.size(); ++i) {
    if (not mesh->template get_cell_map<d-1>(bd_list[i]).is_flat()) flat = false;
  }
  return flat;
}

template<size_t dimension>
Mesh<dimension> * Mesh_builder<dimension>::build(const char * meshfile, const char *mapfile)
{
  Mesh<dimension> * mesh_p = new Mesh<dimension>;
  std::ifstream f(meshfile);
  json data = json::parse(f);

  Maps_loader<dimension>* maps_p = new Maps_loader<dimension>(mapfile);

  assert(data["Dimension"] = dimension && "Mesh builder dimension does not match the dimension specified in file");
  // Embedding
  {
    auto const &map = data["Map"];
    size_t const nb_maps_embed = map["Arguments"].size();
    size_t const nb_maps_metric = map["Arguments_metric"].size();
    size_t const outer_dim = map["Outer_dimension"];
    assert(outer_dim == 3 && "Only embeddings in dimension 3 are implemented yet");
    for (size_t i = 0; i < nb_maps_embed; ++i) {
      if (outer_dim == 3) {
        if (map["Arguments"][i].is_null()) {
          mesh_p->_maps.emplace_back(maps_p->get_new_embedding_3D(i));
        } else {
          std::vector<double> extra = map["Arguments"][i];
          mesh_p->_maps.emplace_back(maps_p->get_new_embedding_3D(i,extra));
        }
      }
    }
    for (size_t i = 0; i < nb_maps_metric; ++i) {
      if (map["Arguments_metric"][i].is_null()) {
        mesh_p->_metric_maps.emplace_back(maps_p->get_new_metrics(i));
      } else {
        std::vector<double> extra = map["Arguments_metric"][i];
        mesh_p->_metric_maps.emplace_back(maps_p->get_new_metrics(i,extra));
      }
    }
  }

  // Vertex 
  auto const &vertices = data["Cells"][0];
  mesh_p->_cells_graph[0].reserve(vertices.size());
  mesh_p->_geo0.reserve(vertices.size());
  for (size_t i = 0; i < vertices.size(); ++i) {
    // Parse graph structure
    std::vector<size_t> map_ids = vertices[i]["Map_ids"];
    mesh_p->_cells_graph[0].emplace_back(i,map_ids);
    
    // Parse geometric structure
    auto const &coords = vertices[i]["Location"];
    std::vector<Eigen::Vector<double,dimension>> coord_vec;
    for (size_t j = 0; j < coords.size(); ++j) {
      assert(coords[j].size() == dimension);
      Eigen::Vector<double,dimension> tmp;
      for (size_t k = 0; k < dimension; ++k) {
        tmp(k) = coords[j][k];
      }
      coord_vec.push_back(tmp);
    }
    mesh_p->_geo0.emplace_back(coord_vec);
  }
  // Higher cells
    // Parse graph structure
  for (size_t d = 1; d <= dimension; ++ d) {
    auto dcells = data["Cells"][d];
    mesh_p->_cells_graph[d].reserve(dcells.size());
    for (size_t i = 0; i < dcells.size(); ++i) {
      std::vector<size_t> map_ids = dcells[i]["Map_ids"];
      std::vector<size_t> bd = dcells[i]["Boundary"];
      mesh_p->_cells_graph[d].emplace_back(mesh_p->_cells_graph,d,i,map_ids,bd);
    }
  }
    // Parse geometric structure
  if constexpr (dimension > 0) { // Edges
    auto dcells = data["Cells"][1];
    mesh_p->_geo1.reserve(dcells.size());
    for (size_t iT = 0; iT < dcells.size(); ++iT) {
      if (dcells[iT]["Mappings"].size() == 1 && dcells[iT]["Mappings"][0] == 0) { // Flat edge
        std::vector<size_t> const & vboundary = mesh_p->get_boundary(0,1,iT);
        std::vector<size_t> const & vboundary_relmap = mesh_p->get_relative_map(0,1,iT);
        assert(vboundary.size() == 2);
        Eigen::Vector<double,dimension> const & v1 = mesh_p->template get_cell_map<0>(vboundary[0]).coord(vboundary_relmap[0]);
        Eigen::Vector<double,dimension> const & v2 = mesh_p->template get_cell_map<0>(vboundary[1]).coord(vboundary_relmap[1]);
        Eigen::Vector<double,dimension> center = (v1+v2)*0.5;
        double diam = (v2-v1).norm();
        Eigen::Vector<double,dimension> map = (v2-v1)/diam;
        std::vector<Simplex<1>> triangulation;
        triangulation.emplace_back(Simplex<1>{Eigen::Vector<double,1>{-0.5},Eigen::Vector<double,1>{0.5}});
        mesh_p->_geo1.emplace_back(center,map,diam,triangulation);
      } else { // Non flat
        std::vector<std::unique_ptr<ParametrizedMap<dimension,1>>> maps;
        std::vector<std::unique_ptr<ParametrizedDerivedMap<dimension,1>>> pullback_maps;
        std::vector<Simplex<1>> triangulation;
        // Fetch mappings
        std::vector<size_t> const & mappings = dcells[iT]["Mappings"];
        const size_t nb_maps = mappings.size();
        const bool mappings_args_null = dcells[iT]["Mappings_extra_args"].is_null();
        const bool pullbacks_args_null = dcells[iT]["Pullbacks_extra_args"].is_null();
        assert(dcells[iT]["Map_ids"].size() == nb_maps);
        assert(mappings_args_null || dcells[iT]["Mappings_extra_args"].size() == nb_maps);
        assert(pullbacks_args_null || dcells[iT]["Pullbacks_extra_args"].size() == nb_maps);
        for (size_t i = 0; i < nb_maps; ++i) {
          if (mappings_args_null || dcells[iT]["Mappings_extra_args"][i].is_null()) {
            maps.emplace_back(maps_p->get_new_edge_map(mappings[i]));
            if (pullbacks_args_null || dcells[iT]["Pullbacks_extra_args"].is_null()) {
              pullback_maps.emplace_back(maps_p->get_new_edge_pullbacks(mappings[i]));
            } else {
              std::vector<double> extra2 = dcells[iT]["Pullbacks_extra_args"][i];
              pullback_maps.emplace_back(maps_p->get_new_edge_pullbacks(mappings[i],extra2));
            }
          } else {
            std::vector<double> extra = dcells[iT]["Mappings_extra_args"][i];
            maps.emplace_back(maps_p->get_new_edge_map(mappings[i],extra));
            if (pullbacks_args_null || dcells[iT]["Pullbacks_extra_args"].is_null()) {
              pullback_maps.emplace_back(maps_p->get_new_edge_pullbacks(mappings[i],extra));
            } else { // Pullbacks extra args override the mappings extra args when present
              std::vector<double> extra2 = dcells[iT]["Pullbacks_extra_args"][i];
              pullback_maps.emplace_back(maps_p->get_new_edge_pullbacks(mappings[i],extra2));
            }
          }
        }
        // Fetch reference elements
        {
          assert(not dcells[iT]["Ref_elem"].is_null() && "Reference element must be provided for custom mappings");
          auto const & ref_elem = dcells[iT]["Ref_elem"];
          assert(ref_elem.size() == 1 && ref_elem[0].size() == 2); // TODO Lift the restriction of only 1 ref element (to allows piecewise defined functions
          double t1 = ref_elem[0][0][0];
          double t2 = ref_elem[0][1][0];
          triangulation.emplace_back(Simplex<1>{Eigen::Vector<double,1>{t1},Eigen::Vector<double,1>{t2}});
        }
        mesh_p->_geo1.emplace_back(maps,pullback_maps,triangulation);
      }
    }
  }
  if constexpr (dimension > 1) { // Faces
    constexpr size_t d = 2;
    auto dcells = data["Cells"][d];
    mesh_p->_geo2.reserve(dcells.size());
    for (size_t iT = 0; iT < dcells.size(); ++iT) {
      std::vector<size_t> const & c1boundary = mesh_p->get_boundary(d-1,d,iT);
      if (dcells[iT]["Mappings"].size() == 1 && dcells[iT]["Mappings"][0] == 0 
          && is_boundary_flat<dimension,d>(mesh_p,c1boundary)) { // Flat face
        std::vector<size_t> const & vboundary = mesh_p->get_boundary(0,d,iT);
        std::vector<size_t> const & vboundary_relmap = mesh_p->get_relative_map(0,d,iT);
        // Collect all vertices
        std::vector<Eigen::Vector<double,dimension>> vertices;
        vertices.reserve(vboundary.size());
        for(size_t i = 0; i < vboundary.size(); ++i) {
          vertices.push_back(mesh_p->template get_cell_map<0>(vboundary[i]).coord(vboundary_relmap[i]));
        }
        // Compute basic geometric data
        double diameter = Geometry::diameter(vertices);
        Eigen::Matrix<double,dimension,d> tangent;
        if constexpr(dimension == d) {
          tangent = Eigen::Matrix<double,dimension,d>::Identity();
        } else {
          tangent = Geometry::tangent_space<dimension,d>(vertices);
        }
        // Compute triangulation
        std::vector<Simplex<dimension>> triangulation_outer;
        if (dcells[iT]["Ref_elem"] == nullptr) { // Build the triangulation ourself, we assume the f star-shaped w.r.t. the mean center
          if (vertices.size() == 3) { // Already a triangle
            triangulation_outer.emplace_back(Simplex<dimension>{vertices[0].transpose(),
                                        vertices[1].transpose(),
                                        vertices[2].transpose()});
          } else {
            // Compute first approximation of the center
            Eigen::Vector<double,dimension> center_l1 = Eigen::Vector<double,dimension>::Zero();
            for (size_t i = 0; i < vertices.size(); ++i) {
              center_l1 += vertices[i];
            }
            center_l1 /= vertices.size();
            std::vector<size_t> const & c1_rel = mesh_p->get_relative_map(d-1,d,iT);
            // Triangulate with respect to this center
            for (size_t i = 0; i < c1boundary.size(); ++i) {
              const auto & edge = mesh_p->template get_cell_map<1>(c1boundary[i]);
              triangulation_outer.emplace_back(Simplex<dimension>{center_l1,
                  edge.evaluate_I(c1_rel[i],edge.get_reference_elem()[0][0]).transpose(),
                  edge.evaluate_I(c1_rel[i],edge.get_reference_elem()[0][1]).transpose()});
            }
          }
        } else { // Use a custom triangulation
          // Note: for the flat case, we want the triangulation with the global coordinate (and not the reference element)
          auto const & ref_elem = dcells[iT]["Ref_elem"];
          for (size_t iS = 0; iS < ref_elem.size(); iS++){
            assert(ref_elem[iS].size() == d+1 && ref_elem[iS][0].size()==dimension 
                                              && ref_elem[iS][1].size()==dimension 
                                              && ref_elem[iS][2].size()==dimension);
            Eigen::Vector<double,d> v1,v2,v3;
            for (size_t i = 0; i < dimension; ++i) {
              v1(i) = ref_elem[iS][0][i];
              v2(i) = ref_elem[iS][1][i];
              v3(i) = ref_elem[iS][2][i];
            }
            triangulation_outer.emplace_back(Simplex<d>{v1,v2,v3});
          }
        }
        // compute the actual center
        Eigen::Vector<double,dimension> center = Eigen::Vector<double,dimension>::Zero();
        double total_vol = 0.;
        for (size_t i = 0; i < triangulation_outer.size(); ++i) {
          const auto & tri = triangulation_outer[i];
          double vol = Geometry::volume_triangle(tri[0],tri[1],tri[2]);
          Eigen::Vector<double,d> loc_center = (tri[0]+tri[1]+tri[2])/3.;
          center += vol*loc_center;
          total_vol += vol;
        }
        center /= total_vol;
        // project the triangulation into the local coordinates
        std::vector<Simplex<d>> triangulation;
        for (size_t i = 0; i < triangulation_outer.size(); ++i) {
          triangulation.emplace_back(
              Simplex<d>({(triangulation_outer[i][0]-center).transpose()*tangent/diameter,
                          (triangulation_outer[i][1]-center).transpose()*tangent/diameter,
                          (triangulation_outer[i][2]-center).transpose()*tangent/diameter}));

        }

        mesh_p->_geo2.emplace_back(center,tangent,diameter,triangulation);
      } else { // Non flat
        std::vector<std::unique_ptr<ParametrizedMap<dimension,d>>> maps;
        std::vector<std::unique_ptr<ParametrizedDerivedMap<dimension,d>>> pullback_maps;
        std::vector<Simplex<d>> triangulation;
        // Fetch mappings
        std::vector<size_t> const & mappings = dcells[iT]["Mappings"];
        const size_t nb_maps = mappings.size();
        const bool mappings_args_null = dcells[iT]["Mappings_extra_args"].is_null();
        const bool pullbacks_args_null = dcells[iT]["Pullbacks_extra_args"].is_null();
        assert(dcells[iT]["Map_ids"].size() == nb_maps);
        assert(mappings_args_null || dcells[iT]["Mappings_extra_args"].size() == nb_maps);
        assert(pullbacks_args_null || dcells[iT]["Pullbacks_extra_args"].size() == nb_maps);
        for (size_t i = 0; i < nb_maps; ++i) {
          if (mappings_args_null || dcells[iT]["Mappings_extra_args"][i].is_null()) {
            maps.emplace_back(maps_p->get_new_face_map(mappings[i]));
            if (pullbacks_args_null || dcells[iT]["Pullbacks_extra_args"].is_null()) {
              pullback_maps.emplace_back(maps_p->get_new_face_pullbacks(mappings[i]));
            } else {
              std::vector<double> extra2 = dcells[iT]["Pullbacks_extra_args"][i];
              pullback_maps.emplace_back(maps_p->get_new_face_pullbacks(mappings[i],extra2));
            }
          } else {
            std::vector<double> extra = dcells[iT]["Mappings_extra_args"][i];
            maps.emplace_back(maps_p->get_new_face_map(mappings[i],extra));
            if (pullbacks_args_null || dcells[iT]["Pullbacks_extra_args"].is_null()) {
              pullback_maps.emplace_back(maps_p->get_new_face_pullbacks(mappings[i],extra));
            } else { // Pullbacks extra args override the mappings extra args when present
              std::vector<double> extra2 = dcells[iT]["Pullbacks_extra_args"][i];
              pullback_maps.emplace_back(maps_p->get_new_face_pullbacks(mappings[i],extra2));
            }
          }
        }
        // Fetch reference element
        {
          assert(not dcells[iT]["Ref_elem"].is_null() && "Reference element must be provided for custom mappings");
          auto const & ref_elem = dcells[iT]["Ref_elem"];
          for (size_t iS = 0; iS < ref_elem.size(); iS++){
            assert(ref_elem[iS].size() == d+1 && ref_elem[iS][0].size()==d 
                                              && ref_elem[iS][1].size()==d 
                                              && ref_elem[iS][2].size()==d);
            Eigen::Vector<double,d> v1(ref_elem[iS][0][0],ref_elem[iS][0][1])
                                   ,v2(ref_elem[iS][1][0],ref_elem[iS][1][1])
                                   ,v3(ref_elem[iS][2][0],ref_elem[iS][2][1]);
            triangulation.emplace_back(Simplex<d>{v1,v2,v3});
          }
        }
        // Insert new cell
        mesh_p->_geo2.emplace_back(maps,pullback_maps,triangulation);
      }
    }
  }
  static_assert(dimension < 3 && "Higher dimension not yet implemented");

  mesh_p->_loader_ref = std::move(std::unique_ptr<Maps_loader<dimension>>(maps_p));
  return mesh_p;
}

#include "preprocessor.hpp"
#define PRED(x, ...) COMPL(IS_1(x))
#define OP(x, ...) template class Manicore::Mesh_builder<x>;
#define CONT(x, ...) DEC(x), __VA_ARGS__

EVAL(WHILE(PRED,OP,CONT,MAX_DIMENSION))

