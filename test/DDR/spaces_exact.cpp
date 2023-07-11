#include <fstream>
#include <iostream>

#include "../testhelpers.hpp"

#include "ddr_spaces.hpp"

#include "mesh_builder.hpp"

using namespace Manicore;

const char *meshfile = "../meshes/test/58_pts.json";
const char *mapfile = "meshes/test/libdisk_maps.so";

int main(int argc, char *argv[]) {

  Err_p err_p;
  NullStream os;
  constexpr size_t dimension = 2;

  // Build the mesh
  std::unique_ptr<Mesh<dimension>> mesh_ptr(Mesh_builder<dimension>::build(meshfile,mapfile));
  
  char *p;
  size_t i_cell=std::strtol(argv[1],&p,10);

  for (int r = 0; r < 4; ++r) {
    DDR_Spaces<dimension> ddr_spaces(*mesh_ptr,r,true,nullptr,os);
    std::cout<<"Degree "<<r<<std::endl;

    for (size_t d = 1; d <= dimension; ++d) {
      std::vector<size_t> boundaries = mesh_ptr->get_boundary(d,dimension,i_cell);
      for (size_t i : boundaries) {
        std::vector<int> space_dim, kernel_dim, range_dim;
        for (size_t k = 0; k < d; ++k) {
          Eigen::MatrixXd diff = ddr_spaces.compose_diff(k,d,i);
          space_dim.push_back(diff.cols());
          range_dim.push_back(diff.colPivHouseholderQr().rank());
          kernel_dim.push_back(space_dim.back()-range_dim.back() - (k > 0? range_dim[k-1] : 0));
          if (k == d-1) {
            space_dim.push_back(diff.rows());
          }
        }
        std::cout<<"Dimension "<<d<<" cell "<<i<<std::endl;
        std::cout<<"Spaces: "<<space_dim[0];
        for (size_t j = 1; j < space_dim.size(); ++j){
          std::cout<<" -> "<<space_dim[j];
        }
        std::cout<<std::endl;
        std::cout<<"Range:  "<<range_dim[0];
        for (size_t j = 1; j < range_dim.size(); ++j){
          std::cout<<" -> "<<range_dim[j];
        }
        std::cout<<" -> 0"<<std::endl;
        std::cout<<"Kernel: "<<kernel_dim[0];
        if (kernel_dim[0] != 1) err_p++;
        for (size_t j = 1; j < kernel_dim.size(); ++j){
          std::cout<<" -> "<<kernel_dim[j];
          if (kernel_dim[j] != 0) err_p++;
        }
        std::cout<<" -> 0"<<err_p<<std::endl;
      }
    }
  }

  std::cout<<"Number of unexpected results: "<< err_p._count<<std::endl;

  return err_p._count;
}
