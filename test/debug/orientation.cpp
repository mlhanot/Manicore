#include "mesh_builder.hpp"
#include "integral.hpp"

#include <iostream>
#include <fstream>
#include <memory>

using namespace Manicore; 

const char *meshfile = "../meshes/test/58_pts.json";
const char *mapfile = "meshes/test/libdisk_maps.so";


int main() {

  std::unique_ptr<Mesh<2>> mesh_p(Mesh_builder<2>::build(meshfile,mapfile));

  constexpr size_t d = 2;

  std::ofstream plotfile("plot_orientation");
  std::ofstream plotfile_ref("plot_orientation_ref");
  std::ofstream plotfile_r("plot_orientation_r");
  std::ofstream plotfile_ref_r("plot_orientation_ref_r");
  for (size_t i_c = 0; i_c < mesh_p->n_cells(d); ++i_c) {
    auto boundary = mesh_p->get_boundary(d-1,d,i_c);
    auto b_rel_maps = mesh_p->get_relative_map(d-1,d,i_c);
    auto const & F = mesh_p->get_cell_map<d>(i_c);
    for (size_t i_bd = 0; i_bd < boundary.size(); ++i_bd) {
      auto const & E = mesh_p->get_cell_map<d-1>(boundary[i_bd]);
      auto x = Geometry::middleSimplex<d-1>(E.get_reference_elem()[0]);
      Eigen::Vector<double,d> pM = E.evaluate_DI(b_rel_maps[i_bd],x);
      auto Ix = E.evaluate_I(b_rel_maps[i_bd],x);
      int orientation = F.get_orientation(0,Ix,pM);
      Eigen::Vector<double,d> tnE;
      auto JIx = F.evaluate_J(0,Ix);
      for (auto const &S : F.get_reference_elem()) {
        if (Geometry::inside<d>(JIx,S)) {
          auto mid = Geometry::middleSimplex<d>(S);
          auto DI = F.evaluate_DI(0,JIx);
          tnE = DI*(JIx - mid);
          break;
        }
      }
      if (mesh_p->get_map_ids(d,i_c)[0] == 0) {
        plotfile << Ix.transpose() <<"\t"<< (orientation*pM.normalized()).transpose()<<"\t"<<i_c<<std::endl;
        plotfile_ref << Ix.transpose() <<"\t"<< pM.normalized().transpose()<<"\t"<<i_c<<std::endl;
        plotfile_ref << Ix.transpose() <<"\t"<< tnE.normalized().transpose()<<"\t"<<i_c<<std::endl;
      } else {
        plotfile_r << Ix.transpose() <<"\t"<< (orientation*pM.normalized()).transpose()<<"\t"<<i_c<<std::endl;
        plotfile_ref_r << Ix.transpose() <<"\t"<< pM.normalized().transpose()<<"\t"<<i_c<<std::endl;
        plotfile_ref_r << Ix.transpose() <<"\t"<< tnE.normalized().transpose()<<"\t"<<i_c<<std::endl;
      }
    }
  }

  return 0;
}

