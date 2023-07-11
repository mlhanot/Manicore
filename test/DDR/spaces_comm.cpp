#include <fstream>
#include <iostream>
#include <cstdlib>

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
  size_t i=std::strtol(argv[1],&p,10);

  auto test_comm = [&err_p,&os,&mesh_ptr,i](int r)->void {
    std::cout<<"Test in degree: "<<r<<std::endl;
    // Build DDR PEC
    std::array<int,dimension> dqr = {20,20};
    DDR_Spaces<dimension> ddr_spaces(*mesh_ptr,r,true,&dqr,os);

    // Test lemma 11
    {
    PEC<dimension> ddr(*mesh_ptr,r,true,&dqr,os);
    PEC<dimension> ddr_po(*mesh_ptr,r+1,true,&dqr,os);
    for (size_t d = 2; d <= dimension; ++d) {
      for(size_t k = 0; k < d - 1; ++k) {
        auto as_degrpo_tr = Eigen::KroneckerProduct(
                      Eigen::MatrixXd::Identity(Manicore::Dimension::ExtDim(d-k-2,d-1),Manicore::Dimension::ExtDim(d-k-2,d-1)),
                      Eigen::MatrixXd::Identity(Manicore::Dimension::PolyDim(r+1,d-1),Manicore::Dimension::PolyDim(r,d-1)));
        Eigen::MatrixXd LHS = ddr_po.get_trimmed(d-k-2,d).transpose();
        LHS = LHS*ddr_po.get_diff(d-k-2,d).transpose();
        LHS = LHS*ddr.get_mass(d-k-1,d,i);
        LHS = LHS*ddr_spaces.full_diff(k,d,i);
        Eigen::MatrixXd RHS = Eigen::MatrixXd::Zero(Manicore::Dimension::PLtrimmedDim(r+1,d-k-2,d),ddr_spaces.dofspace(k).dimensionCell(d,i));
        
        std::vector<size_t> boundary = mesh_ptr->get_boundary(d-1,d,i);
        for (size_t j = 0; j < boundary.size(); ++j) {
          Eigen::MatrixXd RHS_f;
          RHS_f = mesh_ptr->get_boundary_orientation(d,i,j)*ddr_po.get_trimmed(d-k-2,d).transpose();
          RHS_f = RHS_f*ddr_po.get_trace(d-k-2,d,i,j).transpose();
          RHS_f = RHS_f*ddr_po.get_mass(d-k-2,d-1,boundary[j]);
          RHS_f = RHS_f*as_degrpo_tr*ddr_spaces.dofspace(k).extendOperator(d-1,d,boundary[j],i,ddr_spaces.full_diff(k,d-1,boundary[j]));
          RHS += RHS_f;
        }
        RHS = (((k+1)%2 == 0)? 1.:-1.)*RHS;
        double err = (LHS - RHS).cwiseAbs().maxCoeff();
        if (err > 1e-8) err_p++;
        std::cout<<"Lemma 11: d: "<<d<<" k: "<<k<<" error: "<<err<<err_p<<std::endl;
      }
    }
    std::cout<<std::endl;
    }

    // Test commutation
    for (size_t k = 1; k <= dimension; ++k) {
      for (size_t d = k; d <= dimension; ++d) {
        double err = (ddr_spaces.potential(k,d,i)*ddr_spaces.compose_diff(k-1,d,i) 
                    - ddr_spaces.full_diff(k-1,d,i)).cwiseAbs().maxCoeff();
        if (err > 1e-8) err_p++;
        std::cout<<"P ud = d: k: "<<k<<" d: "<<d<<" error: "<<err<<err_p<<std::endl;
      }
    }
    std::cout<<std::endl;

    // Test complex
    for (size_t k = 1; k < dimension; ++k) {
      for (size_t d = k+1; d <= dimension; ++d) {
        double err = (ddr_spaces.full_diff(k,d,i)*ddr_spaces.compose_diff(k-1,d,i)).cwiseAbs().maxCoeff();
        if (err > 1e-8) err_p++;
        std::cout<<"d d = 0: k:  "<<k<<" d: "<<d<<" error: "<<err<<err_p<<std::endl;
      }
    }
    std::cout<<std::endl;
  };

  test_comm(0);
  test_comm(1);
  test_comm(2);
  test_comm(3);

  std::cout<<"Number of unexpected results: "<< err_p._count<<std::endl;

  return err_p._count;
}

