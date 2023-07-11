#include <fstream>
#include <iostream>

#include "../testhelpers.hpp"
#include "../testfunctions.hpp"

#include "ddr_spaces.hpp"

#include "mesh_builder.hpp"

using namespace Manicore;

constexpr size_t dimension = 2;
constexpr bool use_thread = false;

const char *meshfile = "../meshes/test/58_pts.json";
//const char *mapfile = "meshes/test/libdisk_maps_debug_flat.so";
const char *mapfileDefault = "meshes/test/libdisk_maps.so";

template<size_t k,size_t d>
using FunctionType = std::function<Eigen::Vector<double,Dimension::ExtDim(k,d)>(const Eigen::Vector<double,dimension> &)>;

template<size_t k, size_t d, size_t dimension>
Eigen::VectorXd l2_proj(const Mesh<dimension>* _mesh, const PEC<dimension> *_ddr, size_t i_cell, int r, 
                           size_t rel_map,
                           FunctionType<k,d> const & func) {
  const int dqr = 20;
  static_assert(1 <= d && d <= dimension);
  if (Dimension::PLtrimmedDim(r,k,d) == 0) return Eigen::VectorXd::Zero(0);
  Integral<dimension,d> integral(_mesh);
  auto const mass = _ddr->get_mass(k,d,i_cell);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(mass.rows());

  auto const quad = integral.generate_quad(i_cell,dqr);
  auto const scalar_quad = integral.evaluate_scalar_quad(i_cell,r,quad);
  auto const volume_quad = integral.evaluate_volume_form(i_cell,quad);
  auto const ext_quad = integral.template evaluate_exterior_quad<k>(i_cell,quad);

  auto const & T = _mesh->template get_cell_map<d>(i_cell);
  for (size_t iqn = 0; iqn < quad.size(); ++iqn) {
    auto const fv = func(T.evaluate_I(rel_map,quad[iqn].vector));
    Eigen::MatrixXd tmp = ext_quad[iqn]*fv;
    b += Eigen::KroneckerProduct(tmp,scalar_quad.row(iqn).transpose())*volume_quad(iqn)*quad[iqn].w;
  }
  Eigen::MatrixXd trimmed = _ddr->get_trimmed(k,d);
  return (trimmed.transpose()*mass*trimmed).ldlt().solve(trimmed.transpose()*b);
};

int main(int argc, char *argv[]) {

  Err_p err_p;
  NullStream os;

  const char * mapfile = (argc > 2)? argv[2] : mapfileDefault;
  // Build the mesh
  std::unique_ptr<Mesh<dimension>> mesh_ptr(Mesh_builder<dimension>::build(meshfile,mapfile));
  
  char *p;
  size_t i_cell=std::strtol(argv[1],&p,10);

  for (int r = 0; r < 5; ++r) {
    DDR_Spaces<dimension> ddr_spaces(*mesh_ptr,r,use_thread,nullptr,os);
    PEC<dimension> _ddr(*mesh_ptr,r,use_thread,nullptr,os);
    std::cout<<"Degree "<<r<<std::endl;

    auto compute_interp = [&]<size_t k,size_t d>(auto const & T,size_t iT, size_t rel_map, 
                                        Eigen::VectorXd const & interp, 
                                        FunctionType<d-k,d> const & origFunction)->double {
      Eigen::VectorXd PI = ddr_spaces.potential(k,d,iT)*ddr_spaces.dofspace(k).restrict(d,iT,interp);
      // Manually evaluate PI
      auto PIfunc = [&](const Eigen::Vector<double,dimension> & x) {
        Eigen::Vector<double,Dimension::ExtDim(d-k,d)> val_poly = Eigen::Vector<double,Dimension::ExtDim(d-k,d)>::Zero();
        for (size_t i_e = 0; i_e < Dimension::ExtDim(d-k,d); ++i_e) {
          for (size_t i_b = 0; i_b < Dimension::PolyDim(r,d); i_b++) {
            val_poly(i_e) += T.evaluate_poly_pullback(rel_map,x,i_b,r)*PI(i_e*Dimension::PolyDim(r,d) + i_b);
          }
        }
        return val_poly;
      };
      Eigen::VectorXd proj_LHS = l2_proj<d-k,d,dimension>(mesh_ptr.get(), &_ddr, iT, r, rel_map, PIfunc);
      Eigen::VectorXd proj_RHS = l2_proj<d-k,d,dimension>(mesh_ptr.get(), &_ddr, iT, r, rel_map, origFunction);

      return (proj_LHS-proj_RHS).dot(proj_LHS-proj_RHS);
    };

    auto compute_diff = [&]<size_t k,size_t d>(auto const & T,size_t iT, size_t rel_map,
                                        Eigen::VectorXd const & interp, 
                                        FunctionType<d-k-1,d> const & origDiff)->double {
      Eigen::VectorXd DI = ddr_spaces.full_diff(k,d,iT)*ddr_spaces.dofspace(k).restrict(d,iT,interp);
      // Manually evaluate PI
      auto PIfunc = [&](const Eigen::Vector<double,dimension> & x) {
        Eigen::Vector<double,Dimension::ExtDim(d-k-1,d)> val_poly = Eigen::Vector<double,Dimension::ExtDim(d-k-1,d)>::Zero();
        for (size_t i_e = 0; i_e < Dimension::ExtDim(d-k-1,d); ++i_e) {
          for (size_t i_b = 0; i_b < Dimension::PolyDim(r,d); i_b++) {
            val_poly(i_e) += T.evaluate_poly_pullback(rel_map,x,i_b,r)*DI(i_e*Dimension::PolyDim(r,d) + i_b);
          }
        }
        return val_poly;
      };
      Eigen::VectorXd proj_LHS = l2_proj<d-k-1,d,dimension>(mesh_ptr.get(), &_ddr, iT, r, rel_map, PIfunc);
      Eigen::VectorXd proj_RHS = l2_proj<d-k-1,d,dimension>(mesh_ptr.get(), &_ddr, iT, r, rel_map, origDiff);

      return (proj_LHS-proj_RHS).dot(proj_LHS-proj_RHS);
    };

    Poly_One poly{r}, poly_po{r+1}; //, poly_p2{r+2};
    Poly_One_trimmed trpoly_po{r+1}, trpoly_p2{r+2};
    auto const & T = mesh_ptr->template get_cell_map<dimension>(i_cell);
    PolyPullback poly_T(poly,T), poly_po_T(poly_po,T);
    PolyPullback trpoly_po_T(trpoly_po,T), trpoly_p2_T(trpoly_p2,T);

    auto test_interpolate = [&]<size_t _d>(auto && test_interpolate) {
      std::vector<size_t> boundaries = mesh_ptr->get_boundary(_d,dimension,i_cell);
      for (size_t i_bd = 0; i_bd < boundaries.size(); i_bd++) {
        std::cout<<"Dimension "<<_d<<" cell "<<boundaries[i_bd]<<std::endl;
        auto const & F = mesh_ptr->template get_cell_map<_d>(boundaries[i_bd]);
        size_t const rel_map = (_d < dimension)? mesh_ptr->get_relative_map(_d,dimension,i_cell)[i_bd] : 0;

        auto test_interpolate_k = [&]<size_t _k>(auto && test_interpolate_k) {
          // Function warpers
          auto F_poly_T = [&](size_t m,const Eigen::Vector<double,dimension> &x)->Eigen::Vector<double,Manicore::Dimension::ExtDim(_k,dimension)> {
                return poly_T.template P_ev<_k>(x);
              };
          auto F_poly_po_T = [&](size_t m,const Eigen::Vector<double,dimension> &x)->Eigen::Vector<double,Manicore::Dimension::ExtDim(_k,dimension)> {
                return poly_po_T.template P_ev<_k>(x);
              };
          auto F_trpoly_po_T = [&](size_t m,const Eigen::Vector<double,dimension> &x)->Eigen::Vector<double,Manicore::Dimension::ExtDim(_k,dimension)> {
                return trpoly_po_T.template P_ev<_k>(x);
              };
          auto F_trpoly_p2_T = [&](size_t m,const Eigen::Vector<double,dimension> &x)->Eigen::Vector<double,Manicore::Dimension::ExtDim(_k,dimension)> {
                return trpoly_p2_T.template P_ev<_k>(x);
              };
          // Interpolate
          Eigen::VectorXd interp = ddr_spaces.template interpolate<_k>(F_poly_T);
          Eigen::VectorXd interp_po = ddr_spaces.template interpolate<_k>(F_poly_po_T);
          Eigen::VectorXd diff = ddr_spaces.template interpolate<_k>(F_trpoly_po_T);
          Eigen::VectorXd diff_po = ddr_spaces.template interpolate<_k>(F_trpoly_p2_T);

          FunctionType<_d-_k,_d> sF_poly_T = [&](const Eigen::Vector<double,dimension> &x)->Eigen::Vector<double,Manicore::Dimension::ExtDim(_d-_k,_d)> {
                auto const Jx = F.evaluate_J(rel_map,x);
                auto const trF = F.template evaluate_DI_p<_k>(rel_map,Jx);
                auto const hodge = mesh_ptr->template getHodge<_k,_d>(boundaries[i_bd],Jx);
                return hodge*trF*poly_T.template P_ev<_k>(x);
              };
          FunctionType<_d-_k,_d> sF_poly_po_T = [&](const Eigen::Vector<double,dimension> &x)->Eigen::Vector<double,Manicore::Dimension::ExtDim(_d-_k,_d)> {
                auto const Jx = F.evaluate_J(rel_map,x);
                auto const trF = F.template evaluate_DI_p<_k>(rel_map,Jx);
                auto const hodge = mesh_ptr->template getHodge<_k,_d>(boundaries[i_bd],Jx);
                return hodge*trF*poly_po_T.template P_ev<_k>(x);
              };
          std::cout<<"Form degree: "<<_k<<std::endl;
          std::cout<<"Test for potential"<<std::endl;
          double err = compute_interp.template operator()<_k,_d>(F,boundaries[i_bd],rel_map,interp,sF_poly_T);
          if (err > 1e-8) err_p++;
          std::cout<<"Error for P^r:     "<<err<<err_p<<std::endl;
          err = compute_interp.template operator()<_k,_d>(F,boundaries[i_bd],rel_map,interp_po,sF_poly_po_T);
          if (err > 1e-8) err_p++;
          std::cout<<"Error for P^{r+1}: "<<err<<err_p<<std::endl;
          if constexpr (_k < _d) {
            FunctionType<_d-_k-1,_d> sdF_trpoly_po_T = [&](const Eigen::Vector<double,dimension> &x)->Eigen::Vector<double,Manicore::Dimension::ExtDim(_d-_k-1,_d)> {
                  auto const Jx = F.evaluate_J(rel_map,x);
                  auto const trF = F.template evaluate_DI_p<_k+1>(rel_map,Jx);
                  auto const hodge = mesh_ptr->template getHodge<_k+1,_d>(boundaries[i_bd],Jx);
                  return hodge*trF*trpoly_po_T.template D_ev<_k>(x);
                };
            FunctionType<_d-_k-1,_d> sdF_trpoly_p2_T = [&](const Eigen::Vector<double,dimension> &x)->Eigen::Vector<double,Manicore::Dimension::ExtDim(_d-_k-1,_d)> {
                  auto const Jx = F.evaluate_J(rel_map,x);
                  auto const trF = F.template evaluate_DI_p<_k+1>(rel_map,Jx);
                  auto const hodge = mesh_ptr->template getHodge<_k+1,_d>(boundaries[i_bd],Jx);
                  return hodge*trF*trpoly_p2_T.template D_ev<_k>(x);
                };
            std::cout<<"Test for differential"<<std::endl;
            err = compute_diff.template operator()<_k,_d>(F,boundaries[i_bd],rel_map,diff,sdF_trpoly_po_T);
            if (err > 1e-8) err_p++;
            std::cout<<"Error for P^{-,r+1}: "<<err<<err_p<<std::endl;
            err = compute_diff.template operator()<_k,_d>(F,boundaries[i_bd],rel_map,diff_po,sdF_trpoly_p2_T);
            if (err > 1e-8) err_p++;
            std::cout<<"Error for P^{-,r+2}: "<<err<<err_p<<std::endl;
          }

          if constexpr(_k < _d) test_interpolate_k.template operator()<_k+1>(test_interpolate_k);
        };
        test_interpolate_k.template operator()<0>(test_interpolate_k);
        std::cout<<std::endl;
      }

      if constexpr(_d < dimension) test_interpolate.template operator()<_d+1>(test_interpolate);
    };
    test_interpolate.template operator()<1>(test_interpolate);
  }

  std::cout<<"Number of unexpected results: "<< err_p._count<<std::endl;
  return err_p._count;
}

