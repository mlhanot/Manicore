/* 
 * Copyright (c) 2023 Marien Hanot <marien-lorenzo.hanot@umontpellier.fr>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *  
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

// Provide the operator needed in DDR-PEC.
// Specifically, it provides the interpolator, the potential and the differential operator.
//

#include "ddr_spaces.hpp"

#include "parallel_for.hpp"

using namespace Manicore;

template<size_t dimension>
DDR_Spaces<dimension>::DDR_Spaces(Mesh<dimension> const & mesh, int r, bool use_threads, std::array<int,dimension> const * dqr_p, std::ostream & output) 
  : _mesh(&mesh), _r(r), _use_threads(use_threads), 
    _ddr(std::make_unique<PEC<dimension>>(mesh,_r,_use_threads,dqr_p,output)),
    _ddr_po(std::make_unique<PEC<dimension>>(mesh,_r+1,_use_threads,dqr_p,output))
  {
    for (size_t i_k = 0; i_k <= dimension; ++i_k) {
      std::array<size_t,dimension+1> ldofs;
      for (size_t i_d = 0; i_d <= dimension; i_d++) {
        ldofs[i_d] = (i_d - i_k >= 0)? Dimension::PLtrimmedDim(_r,i_d - i_k, i_d) : 0;
      }
      _dofspace[i_k] = DOFSpace(_mesh,ldofs);
    }
    // Resize ops
    for (size_t i_d = 0; i_d <= dimension; i_d++) {
      _ops[i_d].resize(_mesh->n_cells(i_d));
    }

    // Init P for d = k
    std::function<void(size_t,size_t,size_t)> init_P0_gen =
      [this](size_t start,size_t end,size_t k)->void {
        for (size_t i = start; i < end; ++i) {
          _ops[k][i].P[k] = Eigen::MatrixXd::Identity(Dimension::PolyDim(_r,k),Dimension::PolyDim(_r,k));
        }
      };
    // Init d for d >= k + 1
    std::function<void(size_t,size_t,size_t,size_t)> init_d_gen =
      [this](size_t start,size_t end,size_t k,size_t d)->void { // d^k_{r,f(d)}
        assert(d > 0 && "init_d_gen call with d = 0");
        assert(d-k>0 && "init_d_gen called with d-k <= 0");
        for (size_t i = start; i < end; ++i) {
          auto mass_mkmo = _ddr->get_mass(d-k-1,d,i);
          auto mass_mk = _ddr->get_mass(d-k,d,i);
          size_t dofdim = _dofspace[k].dimensionCell(d,i);
          std::vector<size_t> boundary = _mesh->get_boundary(d-1,d,i);
          // RHS
          Eigen::MatrixXd RHS = Eigen::MatrixXd::Zero(Dimension::PLDim(_r,d-k-1,d),dofdim);
          // (-1)^k+1 \int_f w_f ^ du_f
          if (Dimension::PLtrimmedDim(_r,d-k,d) > 0) {
            RHS.rightCols(Dimension::PLtrimmedDim(_r,d-k,d)) = (((k+1)%2 == 0)? 1. : -1.) *_ddr->get_diff_as_degr(d-k-1,d).transpose()*mass_mk*_ddr->get_trimmed(d-k,d);
          }
          // Sum over f
          for (size_t j = 0; j < boundary.size(); ++j) {
            if (d==1) {
              RHS.col(j) = (j%2==0?-1.:1.)*_ddr->get_trace(d-k-1,d,i,j).transpose();
            } else {
              RHS += _mesh->get_boundary_orientation(d,i,j)*
                    _ddr->get_trace(d-k-1,d,i,j).transpose()*_ddr->get_mass(d-k-1,d-1,boundary[j])*
                    _dofspace[k].extendOperator(d-1,d,boundary[j],i,_ops[d-1][boundary[j]].P[k]);
            }
          }
          _ops[d][i].full_diff[k] = mass_mkmo.ldlt().solve(RHS);
          assert(_ops[d][i].full_diff[k].rows() == (int)Dimension::PLDim(_r,d-k-1,d) && 
                 _ops[d][i].full_diff[k].cols() == (int)(_dofspace[k].dimensionCell(d,i))
                                                 && "Wrong dimension for full_diff operator");

        }
      };
    // Init P for d >= k + 1
    std::function<void(size_t,size_t,size_t,size_t)> init_P_gen =
      [this](size_t start,size_t end,size_t k,size_t d)->void { // d^k_{r,f(d)}
        assert(d-k>0 && "init_d_gen called with d-k <= 0");
        for (size_t i = start; i < end; ++i) {
          auto mass_mk = _ddr->get_mass(d-k,d,i);
          auto mass_rpo_dmkmo = _ddr_po->get_mass(d-k-1,d,i);
          size_t dofdim = _dofspace[k].dimensionCell(d,i);
          std::vector<size_t> boundary = _mesh->get_boundary(d-1,d,i);
          auto as_degrpo = Eigen::KroneckerProduct(
                Eigen::MatrixXd::Identity(Dimension::ExtDim(d-k-1,d),Dimension::ExtDim(d-k-1,d)),
                Eigen::MatrixXd::Identity(Dimension::PolyDim(_r+1,d),Dimension::PolyDim(_r,d)));
          const size_t dim_uf = Dimension::kPLDim(_r,d-k,d);
          const size_t dim_vf = Dimension::kPLDim(_r-1,d-k+1,d);
          assert(Dimension::PLDim(_r,d-k,d) == dim_uf+dim_vf && "Wrong size for test functions in P");
          // LHS
          Eigen::MatrixXd LHS(Dimension::PLDim(_r,d-k,d),Dimension::PLDim(_r,d-k,d));
          LHS.topRows(dim_uf) = (_ddr_po->get_diff(d-k-1,d)*_ddr_po->get_reduced_Koszul_m1(d-k,d)).transpose();
          if (dim_vf > 0) {
            LHS.bottomRows(dim_vf) = _ddr->get_reduced_Koszul_m1(d-k+1,d).transpose();
          }
          LHS = LHS*(((k+1)%2==0)? 1.: -1.)* mass_mk;

          // RHS
          Eigen::MatrixXd RHS = Eigen::MatrixXd::Zero(Dimension::PLDim(_r,d-k,d),dofdim);
          RHS.topRows(dim_uf) = _ddr_po->get_reduced_Koszul_m1(d-k,d).transpose()
                               *mass_rpo_dmkmo*as_degrpo*_ops[d][i].full_diff[k];
          if (dim_vf > 0) {
            RHS.bottomRightCorner(dim_vf,Dimension::PLtrimmedDim(_r,d-k,d)) = (((k+1)%2==0)? 1.: -1.) * 
                                  _ddr->get_reduced_Koszul_m1(d-k+1,d).transpose()*
                                  mass_mk*_ddr->get_trimmed(d-k,d);
          }
          for (size_t j = 0; j < boundary.size(); ++j) {
            auto as_degrpo_tr = Eigen::KroneckerProduct(
                Eigen::MatrixXd::Identity(Dimension::ExtDim(d-k-1,d-1),Dimension::ExtDim(d-k-1,d-1)),
                Eigen::MatrixXd::Identity(Dimension::PolyDim(_r+1,d-1),Dimension::PolyDim(_r,d-1)));
            if (d==1) {
              RHS.block(0,j,dim_uf,1) -= (j%2==0?-1.:1.)
                                          *_ddr_po->get_reduced_Koszul_m1(d-k,d).transpose()
                                          *_ddr_po->get_trace(d-k-1,d,i,j).transpose();
            } else {
                RHS.topRows(dim_uf) -= _mesh->get_boundary_orientation(d,i,j)
                 *_ddr_po->get_reduced_Koszul_m1(d-k,d).transpose()
                 *_ddr_po->get_trace(d-k-1,d,i,j).transpose()*_ddr_po->get_mass(d-k-1,d-1,boundary[j])
                 *as_degrpo_tr*_dofspace[k].extendOperator(d-1,d,boundary[j],i,_ops[d-1][boundary[j]].P[k]);
            }
          }
          _ops[d][i].P[k] = LHS.partialPivLu().solve(RHS);
        }
      };

    std::function<void(size_t,size_t,size_t,size_t)> init_projd_gen =
      [this](size_t start,size_t end,size_t k,size_t d)->void { // d^k_{r,f(d)}
        for (size_t i = start; i < end; ++i) {
          assert(d-k > 0 && "Projector called on wrong dimension/forms");
          auto mass_mkmo = _ddr->get_mass(d-k-1,d,i);
          // LHS
          Eigen::MatrixXd trimmed = _ddr->get_trimmed(d-k-1,d);
          Eigen::MatrixXd LHS = trimmed.transpose()*mass_mkmo*trimmed;
          // RHS
          Eigen::MatrixXd RHS = trimmed.transpose()*mass_mkmo*_ops[d][i].full_diff[k];
          _ops[d][i].diff[k] = LHS.ldlt().solve(RHS);
        }
      };

    output<<"[DDR Spaces] Initializing potentials for d=k"<<std::endl;
    for (size_t i = 0; i <= dimension; ++i) {
      parallel_for(_ops[i].size(),std::bind(init_P0_gen,std::placeholders::_1,std::placeholders::_2,i),_use_threads);
    }
    for (size_t i_e = 1; i_e <= dimension; ++i_e) {
      output<<"[DDR Spaces] Initializing diff for d=k+"<<i_e<<std::endl;
      for (size_t i_d = i_e; i_d <= dimension; ++i_d) {
        parallel_for(_ops[i_d].size(),std::bind(init_d_gen,std::placeholders::_1,std::placeholders::_2,i_d-i_e,i_d),_use_threads);
      }
      output<<"[DDR Spaces] Initializing potentials for d=k"<<i_e<<std::endl;
      for (size_t i_d = i_e; i_d <= dimension; ++i_d) {
        parallel_for(_ops[i_d].size(),std::bind(init_P_gen,std::placeholders::_1,std::placeholders::_2,i_d-i_e,i_d),_use_threads);
      }
    }
    // Init diff
    output<<"[DDR Spaces] Initializing ul diff"<<std::endl;
    for (size_t i = 0; i <= dimension; ++i) {
      for (size_t j = 0; j < i; ++j) {
        parallel_for(_ops[i].size(),std::bind(init_projd_gen,std::placeholders::_1,std::placeholders::_2,j,i),_use_threads);
      }
    }
  }

template<size_t dimension>
template<size_t k>
Eigen::VectorXd DDR_Spaces<dimension>::interpolate(FunctionType<k> const & func, 
                                              std::array<int,dimension> const * dqr_p) const 
{
  static_assert(k <= dimension);
  DOFSpace<dimension> const & dofsp = _dofspace[k];
  Eigen::VectorXd qh(dofsp.dimensionMesh());

  std::array<int,dimension> dqr;
  if (dqr_p == nullptr) {
    for (size_t i = 0; i < dimension; ++i){
      dqr[i] = std::min(4*_r + 7,QuadratureMaxDegree[i]);
    }
  } else {
    dqr = *dqr_p;
  }
  // General l2_projection, compute pi^{-,d-k}_{r,f} \star^k func
  auto l2_proj = [&]<size_t d>(const Integral<dimension,d> &integral, size_t i_cell)->Eigen::VectorXd {
    static_assert(1 <= d && d <= dimension);
    if (Dimension::PLtrimmedDim(_r,d-k,d) == 0) return Eigen::VectorXd::Zero(0);
    auto const mass = _ddr->get_mass(d-k,d,i_cell);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(mass.rows());

    auto const quad = integral.generate_quad(i_cell,dqr[d-1]);
    auto const scalar_quad = integral.evaluate_scalar_quad(i_cell,_r,quad);
    auto const volume_quad = integral.evaluate_volume_form(i_cell,quad);
    auto const ext_quad = integral.template evaluate_exterior_quad<d-k>(i_cell,quad);

    auto const & T = _mesh->template get_cell_map<d>(i_cell);
    for (size_t iqn = 0; iqn < quad.size(); ++iqn) {
      auto const fv = func(_mesh->get_map_ids(d,i_cell)[0],T.evaluate_I(0,quad[iqn].vector));
      auto const trT = T.template evaluate_DI_p<k>(0,quad[iqn].vector);
      auto const hodge = _mesh->template getHodge<k,d>(i_cell,quad[iqn].vector);
      Eigen::MatrixXd tmp = ext_quad[iqn]*hodge*trT*fv;
      b += Eigen::KroneckerProduct(tmp,scalar_quad.row(iqn).transpose())*volume_quad(iqn)*quad[iqn].w;
    }
    Eigen::MatrixXd trimmed = _ddr->get_trimmed(d-k,d);
    return (trimmed.transpose()*mass*trimmed).ldlt().solve(trimmed.transpose()*b);
  };

  // Interpolate at cells
  auto interpolate_cells = [&]<size_t d>(size_t start,size_t end)->void{
    for (size_t i_cell = start; i_cell < end; ++i_cell) {
      Integral<dimension,d> integral(_mesh);
      qh.segment(dofsp.globalOffset(d,i_cell),Dimension::PLtrimmedDim(_r,d-k,d)) 
        = l2_proj.template operator()<d>(integral,i_cell);
    }
  };

  auto _itterate_cells = [&]<size_t _d>(auto &&_itterate_cells) {
    parallel_for(_mesh->n_cells(_d),[&](size_t start,size_t end)->void {
        return interpolate_cells.template operator()<_d>(start,end);
        },_use_threads);
    if constexpr(_d < dimension) _itterate_cells.template operator()<_d+1>(_itterate_cells);
  };

  // Interpolate vertices if this is a 0-form
  if constexpr(k == 0) {
    parallel_for(_mesh->n_cells(0),
        [&](size_t start,size_t end)->void{
          assert(dofsp.numLocalDofs(0) == 1 && "Interpolation at vertices only done for scalar field");
          for (size_t iV = start; iV < end; ++iV) {
            qh.segment(dofsp.globalOffset(0,iV),1) = func(_mesh->get_map_ids(0,iV)[0],_mesh->template get_cell_map<0>(iV).coord(0));
          }
        },_use_threads);
  }
  // Interpolate higher dimensional cells starting from k-cells up to dimension-cells
  _itterate_cells.template operator()<std::max<size_t>(1,k)>(_itterate_cells);

  return qh;
}

template<size_t dimension>
Eigen::MatrixXd DDR_Spaces<dimension>::compose_diff(size_t k,size_t d,size_t i_cell) const {
  assert(d <= dimension && k < d && i_cell < _ops[d].size() && "Access of diff out of range");
  assert(_dofspace[k+1].numLocalDofs(0) == 0 && "Expected no dofs on vertices");
  if (d==1) {
    return _ops[1][i_cell].diff[k];
  } else {
    Eigen::MatrixXd diff(_dofspace[k+1].dimensionCell(d,i_cell),_dofspace[k].dimensionCell(d,i_cell));
    for (size_t i_d = 1; i_d < d; ++i_d) {
      if (_dofspace[k+1].numLocalDofs(i_d) > 0) {
        std::vector<size_t> boundary = _mesh->get_boundary(i_d,d,i_cell);
        for (size_t i_bd = 0; i_bd < boundary.size(); ++i_bd) {
          diff.middleRows(_dofspace[k+1].localOffset(i_d,d,i_bd,i_cell),_dofspace[k+1].numLocalDofs(i_d)) =
              _dofspace[k].extendOperator(i_d,d,boundary[i_bd],i_cell,_ops[i_d][boundary[i_bd]].diff[k]);
        }
      }
    }
    if (_dofspace[k+1].numLocalDofs(d) > 0) {
      diff.bottomRows(_dofspace[k+1].numLocalDofs(d)) = _ops[d][i_cell].diff[k];
    }
    return diff;
  }
}

template<size_t dimension>
Eigen::MatrixXd DDR_Spaces<dimension>::computeL2Product(size_t k, size_t d,size_t i_cell) const {
  assert(d <= dimension && k <= d && i_cell < _mesh->n_cells(d) && "Access cell out of range");
  auto const & dofspace = _dofspace[k];
  // Top dimensional
  Eigen::MatrixXd const & P = potential(k,d,i_cell);
  Eigen::MatrixXd rv = P.transpose()*_ddr->get_mass(k,d,i_cell)*P;
  // Depth-first travel to reuse traces matrices
  auto _computeTracesL2 = [&]<size_t _d>(auto && _computeTracesL2, size_t i_bAbs, const Eigen::MatrixXd & trP) {
    // Contribution from this _d-cell
    Eigen::MatrixXd Ptr = dofspace.extendOperator(_d,d,i_bAbs,i_cell,potential(k,_d,i_bAbs));
    if constexpr (_d == 0) {
      rv += (trP - Ptr).transpose()*(trP-Ptr); // The mass is trivial (and cannot be queried from PEC)
      return; // No more boundary
    } else {
      rv += (trP - Ptr).transpose()*_ddr->get_mass(k,_d,i_bAbs)*(trP-Ptr);
      // Contribution from its boundary
      if (_d <= k) return; // dimension of boundary lower than form degree, stop here
      auto const & boundary = _mesh->get_boundary(_d-1,_d,i_bAbs);
      for (size_t i_bb = 0; i_bb < boundary.size(); ++i_bb) {
        Eigen::MatrixXd trPb = _ddr->get_trace(k,_d,i_bAbs,i_bb)*trP;
        _computeTracesL2.template operator()<_d-1>(_computeTracesL2, boundary[i_bb], trPb);
      }
    }
  };
  // _computeTracesL2 must be started with a constant expression
  // _initiateTracesL2 iterate from the top dimension until d
  auto _initiateTracesL2 = [&]<size_t _d>(auto && _initiateTracesL2) {
    if constexpr(_d == 0) { // No more boundary
      return; 
    } else if (k >= d) { // Form degree too high for the boundary of dimension d-1
      return;
    } else {
      if (_d > d) { // dimension started too high, propagate on lower dimension
        _initiateTracesL2.template operator()<_d-1>(_initiateTracesL2);
      } else if (_d == d) { // Initiate for all boundary
        auto const & boundary = _mesh->get_boundary(_d-1,_d,i_cell);
        for (size_t i_b = 0; i_b < boundary.size(); ++i_b) {
          Eigen::MatrixXd trP = _ddr->get_trace(k,_d,i_cell,i_b)*P;
          _computeTracesL2.template operator()<_d-1>(_computeTracesL2, boundary[i_b], trP);
        }
      }
    }
  };
  // Contribution from boundary element
  _initiateTracesL2.template operator()<dimension>(_initiateTracesL2);
  return rv;
}


#include "preprocessor.hpp"
// Instantiate the class for all dimensions
#define PRED(x, ...) COMPL(IS_1(x))
#define OP(x, ...) template class Manicore::DDR_Spaces<x>;
#define CONT(x, ...) DEC(x), __VA_ARGS__
EVAL(WHILE(PRED,OP,CONT,MAX_DIMENSION))
#undef PRED
#undef OP
#undef CONT
// Instantiate the template function interpolate for all degree and dimensions
#define PRED(d, k, ...) COMPL(IS_1(d))
#define OP(d, k, ...) template Eigen::VectorXd Manicore::DDR_Spaces<d>::interpolate<k>(Manicore::DDR_Spaces<d>::FunctionType<k> const &, std::array<int,d> const *) const;
#define CONT(d, k, ...) IF_ELSE_2(IS_0(k))(DEC(d), DEC(d), d,DEC(k)), __VA_ARGS__
EVAL(WHILE(PRED,OP,CONT,MAX_DIMENSION,MAX_DIMENSION))

