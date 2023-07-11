#include <fstream>
#include <iostream>

#include "../testhelpers.hpp"

#include "ddr_spaces.hpp"
#include "pec.hpp"

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
  
  for (int r = 1; r < 4; ++r) {
    DDR_Spaces<dimension> ddr_spaces(*mesh_ptr,r,true,nullptr,os);
    PEC<dimension> _ddr(*mesh_ptr,r,true,nullptr,os);
    std::cout<<"Degree "<<r<<std::endl;

    for (size_t d = 1; d <= dimension; ++d) {
      std::vector<size_t> boundaries = mesh_ptr->get_boundary(d,dimension,i_cell);
      for (size_t i : boundaries) {
        std::cout<<"Dimension "<<d<<" cell "<<i<<std::endl;
        for (size_t k = 0; k <= d; ++k) {
          Eigen::MatrixXd mass_mk = _ddr.get_mass(d-k,d,i);
          // LHS
          Eigen::MatrixXd trimmed = _ddr.get_trimmed(d-k,d);
          Eigen::MatrixXd LHS = trimmed.transpose()*mass_mk*trimmed;
          Eigen::MatrixXd RHS = trimmed.transpose()*mass_mk*ddr_spaces.potential(k,d,i);
          Eigen::MatrixXd piP = LHS.ldlt().solve(RHS);
  //        std::cout<<"form "<<k<<std::endl<<prune(piP)<<std::endl;
          const size_t dimPLt = Manicore::Dimension::PLtrimmedDim(r,d-k,d);
          piP.rightCols(dimPLt) -= Eigen::MatrixXd::Identity(dimPLt,dimPLt);
          double err = piP.cwiseAbs().maxCoeff();
          if (err > 1e-8) err_p++;
          std::cout<<"form "<<k<<" error "<<err<<err_p<<std::endl;
        }
      }
    }
  }

  std::cout<<"Number of unexpected results: "<< err_p._count<<std::endl;

  return err_p._count;
}
