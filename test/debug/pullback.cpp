#include "mesh_builder.hpp"
#include "integral.hpp"

#include <iostream>
#include <fstream>
#include <memory>

#include "exterior_objects.hpp" // To initialize the module

using namespace Manicore; 

const char *meshfile = "../meshes/test/58_pts.json";
const char *mapfile = "meshes/test/libdisk_maps.so";

std::vector<Eigen::Vector<double,1>> locs_on_edges(Mesh<2>* mesh_p, size_t id) {
  std::vector<Eigen::Vector<double,1>> rv;
  auto const & E = mesh_p->get_cell_map<1>(id);
  constexpr int nb_pts = 30;
  constexpr double h = 1./nb_pts;
  auto const & T = E.get_reference_elem()[0];
  for (int ix = 0; ix <=nb_pts; ++ix) {
    double l1 = h*ix;
    rv.push_back(Eigen::Vector<double,1>(l1*T[0]+(1.-l1)*T[1]));
  }
  return rv;
}
std::vector<Eigen::Vector<double,2>> locs_on_faces(Mesh<2>* mesh_p, size_t id) {
  std::vector<Eigen::Vector<double,2>> rv;
  auto const & F = mesh_p->get_cell_map<2>(id);
  constexpr int nb_pts = 10;
  double h = 1./nb_pts;
  auto const & triang = F.get_reference_elem();
  for (const auto & T : triang) {
    for (int ix = 0; ix <= nb_pts; ++ix) {
      for (int iy = 0; iy <= nb_pts; ++iy) {
        double l1 = h*ix; // Barycentric coordinates
        double l2 = h*iy;
        double l3 = 1. - l1 - l2;
        if (l3 > 0.) {
          rv.push_back(l1*T[0]+l2*T[1]+l3*T[2]);
        }
      }
      double l1 = h*ix;
      double l2 = 1. - l1;
      rv.push_back(l1*T[0]+l2*T[1]);
    }
  }
  return rv;
}

int main() {
  constexpr int max_r = 5;
  constexpr int p_basis = 4;
  Initialize_exterior_module<2>::init(max_r);

  std::unique_ptr<Mesh<2>> mesh_p(Mesh_builder<2>::build(meshfile,mapfile));

  std::ofstream ref_file("ref_plot");
  std::ofstream pullback_file("pullback_plot");
  int accE = 0;
  for (size_t i = 0; i < mesh_p->n_cells(1);++i) {
    if(mesh_p->get_map_ids(1,i)[0] == 0) {
      auto const & E = mesh_p->get_cell_map<1>(i);
      auto locs = locs_on_edges(mesh_p.get(),i);
      for(auto const & loc : locs) {
        ref_file<<loc.transpose()<<" 0 "<<E.evaluate_poly_on_ref(loc,p_basis,max_r)<<" "<<i<<std::endl;
        Eigen::Vector2d Iloc = E.evaluate_I(0,loc);
        pullback_file<<Iloc.transpose()<<" "<<E.evaluate_poly_pullback(0,Iloc,p_basis,max_r)<<" "<<i<<std::endl;
        accE++;
      }
    }
  }
  std::cout<<accE<<" points ploted for edges"<<std::endl;
  int accF = 0;
  for (size_t i = 0; i < mesh_p->n_cells(2);++i) {
    if(mesh_p->get_map_ids(1,i)[0] == 0) {
      auto const & F = mesh_p->get_cell_map<2>(i);
      auto locs = locs_on_faces(mesh_p.get(),i);
      for(auto const & loc : locs) {
        ref_file<<loc.transpose()<<" "<<F.evaluate_poly_on_ref(loc,p_basis,max_r)<<" "<<i<<std::endl;
        Eigen::Vector2d Iloc = F.evaluate_I(0,loc);
        pullback_file<<Iloc.transpose()<<" "<<F.evaluate_poly_pullback(0,Iloc,p_basis,max_r)<<" "<<i<<std::endl;
        accF++;
      }
    }
  }
  std::cout<<accF<<" points ploted for faces"<<std::endl;

  return 0;
}

