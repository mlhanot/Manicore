#include "mesh_builder.hpp"

#include <iostream>
#include <fstream>
#include <memory>

using namespace Manicore; 

const char *meshfile = "../meshes/test/58_pts.json";
const char *mapfile = "meshes/test/libdisk_maps.so";

void plot_vertex(Mesh<2>* mesh_p, size_t g_id, std::ofstream &file,std::ofstream &file3,dCell_map<2,0> const &V,size_t m_id,double val) {
  file<<V.coord(m_id).transpose()<<" "<<val<<std::endl;
  file3<<mesh_p->get_3D_embedding(g_id,V.coord(m_id)).transpose()<<" "<<val<<std::endl;
}
void plot_edge(Mesh<2>* mesh_p, size_t g_id, std::ofstream &file,std::ofstream &file3,dCell_map<2,1> const &E,size_t m_id,double val) {
  constexpr int nb_pts = 30;
  constexpr double h = 1./nb_pts;
  auto const & T = E.get_reference_elem()[0];
  for (int ix = 0; ix <=nb_pts; ++ix) {
    double l1 = h*ix;
    file<<E.evaluate_I(m_id,Eigen::Vector<double,1>(l1*T[0]+(1.-l1)*T[1])).transpose()<<" "<<val<<std::endl;
    file3<<mesh_p->get_3D_embedding(g_id,E.evaluate_I(m_id,Eigen::Vector<double,1>(l1*T[0]+(1.-l1)*T[1]))).transpose()<<" "<<val<<std::endl;
  }
}
void plot_face(Mesh<2>* mesh_p, size_t g_id, std::ofstream &file,std::ofstream &file3,dCell_map<2,2> const &F,size_t m_id,double val) {
  constexpr int nb_pts = 5;
  double h = 1./nb_pts;
  auto const & triang = F.get_reference_elem();
  int acc = 0;
  for (const auto & T : triang) {
    for (int ix = 0; ix <= nb_pts; ++ix) {
      for (int iy = 0; iy <= nb_pts; ++iy) {
        double l1 = h*ix; // Barycentric coordinates
        double l2 = h*iy;
        double l3 = 1. - l1 - l2;
        if (l3 > 0.) {
          file<<F.evaluate_I(m_id,l1*T[0]+l2*T[1]+l3*T[2]).transpose()<<" "<<val+acc<<std::endl;
          file3<<mesh_p->get_3D_embedding(g_id,F.evaluate_I(m_id,l1*T[0]+l2*T[1]+l3*T[2])).transpose()<<" "<<val+acc<<std::endl;
  double err = (F.evaluate_J(m_id,F.evaluate_I(m_id,l1*T[0]+l2*T[1]+l3*T[2])) - (l1*T[0]+l2*T[1]+l3*T[2])).norm();
  if (err > 1e-10) {
    std::cout<<"err: "<<err<<std::endl;
  std::cout<<"x: "<<(l1*T[0]+l2*T[1]+l3*T[2]).transpose()<<" I: "<<(F.evaluate_I(m_id,l1*T[0]+l2*T[1]+l3*T[2])).transpose()<<" JI: "<<(F.evaluate_J(m_id,F.evaluate_I(m_id,l1*T[0]+l2*T[1]+l3*T[2]))).transpose()<<std::endl;
  }
        }
      }
      // To cover the edge l3 = 0
      double l1 = h*ix;
      double l2 = 1. - l1;
      file<<F.evaluate_I(m_id,l1*T[0]+l2*T[1]).transpose()<<" "<<val+acc<<std::endl;
      file3<<mesh_p->get_3D_embedding(g_id,F.evaluate_I(m_id,l1*T[0]+l2*T[1])).transpose()<<" "<<val+acc<<std::endl;
    }
    acc += 20;
  }
}

int main() {
  std::unique_ptr<Mesh<2>> mesh_p(Mesh_builder<2>::build(meshfile,mapfile));

  std::cout<<"Nb vertices: "<<mesh_p->n_cells(0)<<std::endl;
  for (size_t d = 1; d <= 2; ++d) {
    std::cout<<"Nb "<<d<<"-cell: "<<mesh_p->n_cells(d)<<std::endl;
/*
    for (size_t i = 0; i < mesh_p->n_cells(d); ++i) {
      for (int k = d-1; k >= 0; --k) {
        auto bd = mesh_p->get_boundary(k,d,i);
        std::cout<<k<<"-Boundary of cell "<<i<<": "<<bd[0];
        for (size_t j = 1; j < bd.size(); ++j) {
          std::cout<<", "<<bd[j];
        }
        std::cout<<std::endl;
        auto bdr = mesh_p->get_relative_map(k,d,i);
        std::cout<<  "     Relative map: "<<i<<": "<<bdr[0];
        for (size_t j = 1; j < bdr.size(); ++j) {
          std::cout<<", "<<bdr[j];
        }
        std::cout<<std::endl;
      }
    }
*/
  }
  std::ofstream plotfile("mesh_plot");
  std::ofstream plotfile3("mesh_plot_3D");
  for (size_t i = 0; i < mesh_p->n_cells(0);++i) {
    plot_vertex(mesh_p.get(),mesh_p->get_map_ids(0,i)[0],plotfile,plotfile3,mesh_p->template get_cell_map<0>(i),0,i);
  }
  for (size_t i = 0; i < mesh_p->n_cells(1);++i) {
    plot_edge(mesh_p.get(),mesh_p->get_map_ids(1,i)[0],plotfile,plotfile3,mesh_p->template get_cell_map<1>(i),0,100.+i);
  }
  for (size_t i = 0; i < mesh_p->n_cells(2);++i) {
    plot_face(mesh_p.get(),mesh_p->get_map_ids(2,i)[0],plotfile,plotfile3,mesh_p->template get_cell_map<2>(i),0,200.);//+i);
  }

  return 0;
}

