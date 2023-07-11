#include "mesh_builder.hpp"
#include "integral.hpp"

#include <iostream>
#include <fstream>
#include <memory>

#include "exterior_objects.hpp" // To initialize the module

using namespace Manicore; 

const char *meshfile = "../meshes/test/58_pts.json";
const char *mapfile = "meshes/test/libdisk_maps.so";

template<size_t d>
double integral_on_dcells(Mesh<2>* mesh_p, size_t id) {
  Integral<2,d> integrator(mesh_p);
  auto quad = integrator.generate_quad(id,12);
  auto quad_scalar = integrator.evaluate_scalar_quad(id,1,quad);
  auto quad_volume = integrator.evaluate_volume_form(id,quad);
  double rv = 0.;
  for (size_t iqr = 0; iqr < quad.size(); ++iqr) {
    rv += quad[iqr].w*quad_scalar(iqr,0)*quad_volume(iqr);
  }
  return rv;
}

int main() {
  constexpr int max_r = 5;
  Initialize_exterior_module<2>::init(max_r);

  std::unique_ptr<Mesh<2>> mesh_p(Mesh_builder<2>::build(meshfile,mapfile));

  {
    double perE = 0.;
    constexpr size_t d = 1;
    for (size_t i = 0; i < mesh_p->n_cells(d);++i) {
      if(mesh_p->get_map_ids(d,i).size() == 2) {
        double tmp = integral_on_dcells<d>(mesh_p.get(),i);
        perE += tmp;
        std::cout<<"Edge "<<i<<" lenght: "<<tmp<<std::endl;
      }
    }
    std::cout<<"Circle length: "<<perE<<std::endl;
  }
  {
    double surfF = 0.;
    constexpr size_t d = 2;
    for (size_t i = 0; i < mesh_p->n_cells(d);++i) {
      if(mesh_p->get_map_ids(d,i)[0] == 0) {
        double tmp = integral_on_dcells<d>(mesh_p.get(),i);
        surfF += tmp; 
        std::cout<<"Face "<<i<<" surface: "<<tmp<<std::endl;
      }
    }
    std::cout<<"Surfaces on half faces: "<<surfF<<std::endl;
    for (size_t i = 0; i < mesh_p->n_cells(d);++i) {
      if(mesh_p->get_map_ids(d,i)[0] == 1) {
        double tmp = integral_on_dcells<d>(mesh_p.get(),i);
        surfF += tmp; 
        std::cout<<"Face "<<i<<" surface: "<<tmp<<std::endl;
      }
    }
    std::cout<<"Surfaces on faces: "<<surfF<<std::endl;
  }

  return 0;
}

