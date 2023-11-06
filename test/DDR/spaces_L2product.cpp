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

    for (size_t d = dimension; d <= dimension; ++d) {
      std::vector<size_t> boundaries = mesh_ptr->get_boundary(d,dimension,i_cell);
      for (size_t i : boundaries) {
        std::cout<<"Dimension "<<d<<" cell "<<i<<std::endl;
        for (size_t k = 0; k <= d; ++k) {
          Eigen::MatrixXd M = ddr_spaces.computeL2Product(k,i);
          if (!M.isApprox(M.transpose())) {
            err_p++;
            std::cout<<"Mass matrix not symmetric"<<err_p<<std::endl;
          } else {
            auto eigenSolver = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(M,Eigen::EigenvaluesOnly);
            bool isSPD = true;
            for (double ev : eigenSolver.eigenvalues()) {
              if (ev < 1e-9) isSPD = false;
 std::cout<<"k: "<<k<<", ev: "<<ev<<std::endl; // TODO DELETE
            }
            if (!isSPD) {
              err_p++;
              std::cout<<"Mass matrix not definite positive"<<err_p<<std::endl;
            }
          }
        }
      }
    }
  }

  std::cout<<"Number of unexpected results: "<< err_p._count<<std::endl;

  return err_p._count;
}
