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

// Provide the general data needed on a mesh.
// Specifically, it provides the generic operators, the mass matrix on each cell, and the trace operators.
//

#include "pec.hpp"

#include "exterior_objects.hpp"
#include "parallel_for.hpp"

using namespace Manicore; 

template<size_t dimension>
PEC<dimension>::PEC(Mesh<dimension> const & mesh,int r, bool use_threads, std::array<int,dimension> const * dqr_p, std::ostream & output) : _r(r) {
  // Initialize global data
  Initialize_exterior_module<dimension>::init(r);
  _dim_table[0] = 0;
  for (size_t i = 1;i <= dimension;++i) {
    _dim_table[i] = _dim_table[i-1] + i+1;
  }
  _list_diff.reserve(_dim_table.back());
  _list_Koszul.reserve(_dim_table.back());
  _list_diff_as_degr.reserve(_dim_table.back());
  _list_trimmed.reserve(_dim_table.back());
  _list_reduced_Koszul_m1.reserve(_dim_table.back());
  _fill_lists<1,0>();

  std::array<size_t,dimension+1> nbelem;
  for (size_t i = 0; i <= dimension; ++i) {
    nbelem[i] = mesh.n_cells(i);
  }

  std::array<int,dimension> dqr;
  if (dqr_p == nullptr) {
    for (size_t i = 0; i < dimension; ++i){
      dqr[i] = std::min(4*_r + 7,QuadratureMaxDegree[i]);
    }
  } else {
    dqr = *dqr_p;
  }
  ///------------------------------------------------------------------------------------------------------------------------------
  // Construct cells, evaluate quad and compute the gram matrix on every elements
  auto evaluate_masses = [this,&mesh,&dqr]<size_t d>(size_t start,size_t end)
  {
    Integral<dimension,d> integral(&mesh);
    for (size_t i_l = start; i_l < end; ++ i_l) {
      _dCellList.template mass<d>()[i_l] = dCell_mass<dimension,d>(i_l,_r,integral.generate_quad(i_l,dqr[d-1]),integral);
    }
  };

  auto compute_masses = [&]<size_t _d>(auto&& compute_masses)
  {
    output<<"[PEC] Constructing "<<_d<<"-Cells masses"<<std::endl;
    _dCellList.template mass<_d>().resize(nbelem[_d]);
    _dCellList.template traces<_d>().resize(nbelem[_d]);
    parallel_for(nbelem[_d],
                 [&](size_t start,size_t end){return evaluate_masses.template operator()<_d>(start,end);}
                 ,use_threads);
    if constexpr(_d < dimension) {
      compute_masses.template operator()<_d+1>(compute_masses);
    }
  };
  compute_masses.template operator()<1>(compute_masses);

  ///------------------------------------------------------------------------------------------------------------------------------
  // Compute traces operator
  auto evaluate_traces_1 = [this,&mesh](size_t start,size_t end)
  {
    for (size_t i_l = start; i_l < end; ++ i_l) {
      _dCellList.template traces<1>()[i_l] = dCell_traces<dimension,1>(i_l,_r,&mesh);
    }
  };
  auto evaluate_traces = [this,&mesh,&dqr]<size_t d>(size_t start,size_t end)
  {
    Integral<dimension,d> integral(&mesh);
    Integral<dimension,d-1> integral_b(&mesh);
    for (size_t i_l = start; i_l < end; ++ i_l) {
      _dCellList.template traces<d>()[i_l] = dCell_traces<dimension,d>(i_l,_r,dqr[d-2],
          _dCellList.template mass<d-1>(), integral, integral_b);
    }
  };

  output<<"[PEC] Constructing "<<1<<"-Cells traces"<<std::endl;
  parallel_for(nbelem[1],evaluate_traces_1,use_threads);
  auto compute_traces = [&]<size_t _d>(auto&& compute_traces)
  {
    output<<"[PEC] Constructing "<<_d<<"-Cells traces"<<std::endl;
    parallel_for(nbelem[_d],
        [&](size_t start,size_t end){return evaluate_traces.template operator()<_d>(start,end);}
        ,use_threads);
    if constexpr(_d < dimension) {
      compute_traces.template operator()<_d+1>(compute_traces);
    }
  };
  compute_traces.template operator()<2>(compute_traces);
}

template<size_t dimension>
template<size_t d,size_t l> void PEC<dimension>::_fill_lists() {
  if constexpr (d <= dimension) {
    if constexpr (l <= d) {
      _list_diff.emplace_back(Diff_full<l,d>::get(_r));
      _list_Koszul.emplace_back(Koszul_full<l,d>::get(_r));
      _list_diff_as_degr.emplace_back(Diff_full<l,d>::get_as_degr(_r));
      { // trimmed
        if constexpr(l == 0) {
          _list_trimmed.emplace_back(Eigen::MatrixXd::Identity(Dimension::PLDim(_r,0,d),Dimension::PLtrimmedDim(_r,0,d)));
        } else if (Dimension::PLtrimmedDim(_r,l,d) == 0) {
          _list_trimmed.emplace_back(Eigen::MatrixXd::Zero(Dimension::PLDim(_r,l,d),Dimension::PLtrimmedDim(_r,l,d)));
        } else {
          Eigen::MatrixXd trimmed(Dimension::PLDim(_r,l,d),Dimension::PLDim(_r,l-1,d)+Dimension::PLDim(_r-1,l+1,d));
          trimmed.block(0,0,Dimension::PLDim(_r,l,d),Dimension::PLDim(_r,l-1,d)) = Diff_full<l-1,d>::get_as_degr(_r);
          if (l < d && _r > 0) {
            trimmed.block(0,Dimension::PLDim(_r,l-1,d),Dimension::PLDim(_r,l,d),Dimension::PLDim(_r-1,l+1,d)) = Koszul_full<l+1,d>::get(_r-1);
          } else {
            assert(Dimension::PLDim(_r-1,l+1,d) == 0);
          }
          Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(trimmed);
          assert(lu_decomp.rank() == (int)Dimension::PLtrimmedDim(_r,l,d));
          _list_trimmed.emplace_back(lu_decomp.image(trimmed));
        }
      }
      if (_r > 0 && l > 0) {
        Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(Koszul_full<l,d>::get(_r-1));
        _list_reduced_Koszul_m1.emplace_back(lu_decomp.image(Koszul_full<l,d>::get(_r-1)));
      } else {
        _list_reduced_Koszul_m1.emplace_back(Eigen::Matrix<double,0,0>::Zero());
      }
      _fill_lists<d,l+1>();
    } else {
       _fill_lists<d+1,0>();
    }
  }
}

template<size_t dimension>
Eigen::MatrixXd PEC<dimension>::get_mass(size_t k, size_t d, size_t i_cell) const
{
  auto fetch_mass = [&]<size_t l>(auto&& fetch_mass) {
    if constexpr(l == dimension) {
      assert(d == l);
      return _dCellList.template mass<l>()[i_cell].masses[k];
    } else {
      if (d == l) {
        return _dCellList.template mass<l>()[i_cell].masses[k];
      } else {
        return fetch_mass.template operator()<l+1>(fetch_mass);
      }
    }
  };
  return fetch_mass.template operator()<1>(fetch_mass);
}

template<size_t dimension>
Eigen::MatrixXd PEC<dimension>::get_trace(size_t k, size_t d, size_t i_cell, size_t j_bd) const
{
  auto fetch_trace = [&]<size_t l>(auto&& fetch_trace) {
    if constexpr(l == dimension) {
      assert(d == l);
      return _dCellList.template traces<l>()[i_cell].traces[k][j_bd];
    } else {
      if (d == l) {
        return _dCellList.template traces<l>()[i_cell].traces[k][j_bd];
      } else {
        return fetch_trace.template operator()<l+1>(fetch_trace);
      }
    }
  };
  return fetch_trace.template operator()<1>(fetch_trace);
}

#include "preprocessor.hpp"
#define PRED(x, ...) COMPL(IS_1(x))
#define OP(x, ...) template class Manicore::PEC<x>;
#define CONT(x, ...) DEC(x), __VA_ARGS__

EVAL(WHILE(PRED,OP,CONT,MAX_DIMENSION))

