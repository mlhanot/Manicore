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

#include "dcell_integral.hpp"

#include <unsupported/Eigen/KroneckerProduct>

using namespace Manicore;

// Doxygen have some trouble with template overloading
/// @cond
template<size_t dimension,size_t d> requires(d > 0 && d <= dimension)
dCell_mass<dimension,d>::dCell_mass(size_t i_cell, int r, const QuadratureRule<d> & quad, const Integral<dimension,d> &integral)
{
  auto const scalar_quad = integral.evaluate_scalar_quad(i_cell,r,quad);
  auto const volume_quad = integral.evaluate_volume_form(i_cell,quad);

  auto compute_masses = [&]<size_t l>(auto&& compute_masses) 
  {
    auto const ext_quad = integral.template evaluate_exterior_quad<l>(i_cell,quad);
    masses[l] = Eigen::MatrixXd::Zero(Dimension::PLDim(r,l,d),Dimension::PLDim(r,l,d));
    for (size_t iqn = 0; iqn < quad.size(); ++iqn) {
      masses[l] += Eigen::KroneckerProduct(ext_quad[iqn],scalar_quad.row(iqn).transpose()*scalar_quad.row(iqn))
                   *volume_quad[iqn]*quad[iqn].w;
    }
    if constexpr(l < d) compute_masses.template operator()<l+1>(compute_masses);
  };
  compute_masses.template operator()<0>(compute_masses);
}

template<size_t dimension,size_t d> requires(d > 0 && d <= dimension)
dCell_traces<dimension,d>::dCell_traces(size_t i_cell, int r, int dqr, 
                                        const std::vector<dCell_mass<dimension,d-1>> & b_masses,
                                        const Integral<dimension,d> &integral, 
                                        const Integral<dimension,d-1> &integral_b) 
{
  auto const mesh = integral.mesh();
  auto const & boundary = mesh->get_boundary(d-1,d,i_cell);
  for (size_t i_bd = 0; i_bd < boundary.size(); ++i_bd){
    auto const quad = integral_b.generate_quad(boundary[i_bd],dqr);

    auto const scalar_b_quad = integral_b.evaluate_scalar_quad(boundary[i_bd],r,quad);
    auto const volume_b_quad = integral_b.evaluate_volume_form(boundary[i_bd],quad);

    auto const scalar_quad = integral.evaluate_scalar_quad_tr(i_cell,i_bd,r,quad);

    auto compute_traces = [&]<size_t l>(auto&& compute_traces)
    {
      Eigen::MatrixXd M_tr = Eigen::MatrixXd::Zero(Dimension::PLDim(r,l,d-1),Dimension::PLDim(r,l,d));
      auto const ext_quad = integral.template evaluate_exterior_quad_tr<l>(i_cell,i_bd,quad);
      for (size_t iqn = 0; iqn < quad.size(); ++iqn) {
        M_tr += Eigen::KroneckerProduct(ext_quad[iqn],scalar_b_quad.row(iqn).transpose()*scalar_quad.row(iqn))
                     *volume_b_quad[iqn]*quad[iqn].w;
      }
      traces[l].emplace_back(b_masses[boundary[i_bd]].masses[l].ldlt().solve(M_tr));
      if constexpr(l < d-1) compute_traces.template operator()<l+1>(compute_traces);
    };
    compute_traces.template operator()<0>(compute_traces);
  }
}

template<size_t dimension> 
dCell_traces<dimension,1>::dCell_traces(size_t i_cell, int r, const Mesh<dimension>* mesh) 
{
  constexpr size_t d = 1;
  auto const & T = mesh->template get_cell_map<d>(i_cell);
  auto const & boundary = mesh->get_boundary(d-1,d,i_cell);
  for (size_t i_bd = 0; i_bd < boundary.size(); ++i_bd){
    size_t bd_map = mesh->get_relative_map(d-1,d,i_cell)[i_bd];
    auto const & V = mesh->template get_cell_map<d-1>(boundary[i_bd]);
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(1,Dimension::PolyDim(r,1));
    for (size_t i_basis = 0; i_basis<Dimension::PolyDim(r,1); ++i_basis) {
      M(i_basis) = T.evaluate_poly_pullback(0,V.coord(bd_map),i_basis,r);
    }
    traces[0].emplace_back(M);
  }
} 
/// @endcond

#include "/home/user/Manicore/include/preprocessor.hpp"
#define PRED(x, y, ...) COMPL(IS_1(x))
#define OP(x, y, ...) template struct Manicore::dCell_mass<x,y>; template struct Manicore::dCell_traces<x,y>;
#define CONT(x, y, ...) IF_ELSE_2(IS_1(y))(DEC(x), DEC(x), x,DEC(y)), __VA_ARGS__
EVAL(WHILE(PRED,OP,CONT,MAX_DIMENSION,MAX_DIMENSION))

