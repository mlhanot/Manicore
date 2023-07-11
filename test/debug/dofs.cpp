#include "mesh_builder.hpp"
#include "dofspace.hpp"

#include <iostream>
#include <memory>

using namespace Manicore; 

const char *meshfile = "../meshes/test/58_pts.json";
const char *mapfile = "meshes/test/libdisk_maps.so";

int main() {
  std::unique_ptr<Mesh<2>> mesh_p(Mesh_builder<2>::build(meshfile,mapfile));

  std::unique_ptr<DOFSpace<2>> dofspace(new DOFSpace<2>(mesh_p.get(),{1,2,3}));

  Eigen::VectorXd one = Eigen::VectorXd::Zero(dofspace->dimensionMesh());
  for (size_t d = 0; d <= 2; ++d) {
    for (size_t i_c = 0; i_c < mesh_p->n_cells(d); ++i_c){
      for (size_t i = 0; i < dofspace->numLocalDofs(d); ++i){
        one(dofspace->globalOffset(d,i_c) + i) = i_c+1000*d;
      }
    }
  }
  std::cout<<"Restriction to vertice 24:\n"<<dofspace->restrict(0,24,one).transpose()<<std::endl;

  std::cout<<"Restriction to edge 0:\n"<<dofspace->restrict(1,0,one).transpose()<<std::endl;
  std::cout<<"Restriction to edge 1:\n"<<dofspace->restrict(1,1,one).transpose()<<std::endl;
  std::cout<<"Restriction to edge 43:\n"<<dofspace->restrict(1,43,one).transpose()<<std::endl;
  std::cout<<"Restriction to edge 97:\n"<<dofspace->restrict(1,97,one).transpose()<<std::endl;
  std::cout<<"Restriction to edge 108:\n"<<dofspace->restrict(1,108,one).transpose()<<std::endl;

  std::cout<<"Restriction to face 0:\n"<<dofspace->restrict(2,0,one).transpose()<<std::endl;
  std::cout<<"Restriction to face 20:\n"<<dofspace->restrict(2,20,one).transpose()<<std::endl;
  std::cout<<"Restriction to face 52:\n"<<dofspace->restrict(2,52,one).transpose()<<std::endl;
  std::cout<<"Restriction to face 19:\n"<<dofspace->restrict(2,19,one).transpose()<<std::endl;
  std::cout<<"Restriction to face 51:\n"<<dofspace->restrict(2,51,one).transpose()<<std::endl;

  std::cout<<std::endl;
  {
    Eigen::MatrixXd onem = dofspace->restrict(0,24,one).transpose();
    std::cout<<"Extension vertice 24 to edge 43:\n"<<dofspace->extendOperator(0,1,24,43,onem)<<std::endl;
    std::cout<<"Extension vertice 24 to edge 97:\n"<<dofspace->extendOperator(0,1,24,97,onem)<<std::endl;
    std::cout<<"Extension vertice 24 to edge 108:\n"<<dofspace->extendOperator(0,1,24,108,onem)<<std::endl;
    std::cout<<"Extension vertice 24 to edge 109:\n"<<dofspace->extendOperator(0,1,24,109,onem)<<std::endl;

    std::cout<<"Extension vertice 24 to face 20:\n"<<dofspace->extendOperator(0,2,24,20,onem)<<std::endl;
    std::cout<<"Extension vertice 24 to face 21:\n"<<dofspace->extendOperator(0,2,24,21,onem)<<std::endl;
    std::cout<<"Extension vertice 24 to face 52:\n"<<dofspace->extendOperator(0,2,24,52,onem)<<std::endl;
    std::cout<<"Extension vertice 24 to face 53:\n"<<dofspace->extendOperator(0,2,24,53,onem)<<std::endl;
  }
  {
    Eigen::MatrixXd onem = dofspace->restrict(1,108,one).transpose();
    std::cout<<"Extension edge 108 to face 20:\n"<<dofspace->extendOperator(1,2,108,20,onem)<<std::endl;
    std::cout<<"Extension edge 108 to face 52:\n"<<dofspace->extendOperator(1,2,108,52,onem)<<std::endl;
  }

  return 0;
}

