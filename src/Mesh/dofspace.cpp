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

#include "dofspace.hpp"

using namespace Manicore;

template<size_t dimension>
Eigen::VectorXd DOFSpace<dimension>::restrict(size_t d, size_t i_cell, const Eigen::VectorXd & vh) const 
{
  Eigen::VectorXd rv = Eigen::VectorXd::Zero(dimensionCell(d,i_cell));
  for (size_t i_d = 0; i_d < d; ++i_d) {
    if (_nb_local_dofs[i_d] > 0) {
      auto const & boundary = _mesh->get_boundary(i_d,d,i_cell);
      for (size_t i_f = 0; i_f < boundary.size(); ++i_f) {
        rv.segment(localOffset(i_d,d,i_f,i_cell), _nb_local_dofs[i_d]) 
          = vh.segment(globalOffset(i_d,boundary[i_f]), _nb_local_dofs[i_d]);
      }
    }
  } // for i_d < d
  if (_nb_local_dofs[d] > 0) {
    rv.tail(_nb_local_dofs[d]) 
      = vh.segment(globalOffset(d,i_cell), _nb_local_dofs[d]);
  }
  return rv;
}


template<size_t dimension>
Eigen::MatrixXd DOFSpace<dimension>::extendOperator(size_t d_boundary, size_t d, size_t i_bd_global, size_t i_cell, const Eigen::MatrixXd & op) const
{
  assert(d_boundary < d);
  Eigen::MatrixXd rv = Eigen::MatrixXd::Zero(op.rows(), dimensionCell(d,i_cell));
  for (size_t i_d = 0; i_d < d_boundary; ++i_d) {
    if (_nb_local_dofs[i_d] > 0) {
      auto const & b_boundary = _mesh->get_boundary(i_d,d_boundary,i_bd_global);
      for (size_t i_V = 0; i_V < b_boundary.size(); ++i_V) {
        rv.middleCols(localOffset(i_d,d,_mesh->global_to_local(i_d,d,b_boundary[i_V],i_cell),i_cell),_nb_local_dofs[i_d]) 
          = op.middleCols(localOffset(i_d,d_boundary,i_V,i_bd_global),_nb_local_dofs[i_d]);
      }
    }
  } // end for i_d < d_boundary
  if (_nb_local_dofs[d_boundary] > 0) {
    rv.middleCols(localOffset(d_boundary,d,_mesh->global_to_local(d_boundary,d,i_bd_global,i_cell),i_cell),_nb_local_dofs[d_boundary])
      = op.middleCols(localOffset(d_boundary,i_bd_global),_nb_local_dofs[d_boundary]);
  }
  return rv;
}

#include "preprocessor.hpp"
#define PRED(x, ...) COMPL(IS_1(x))
#define OP(x, ...) template class Manicore::DOFSpace<x>;
#define CONT(x, ...) DEC(x), __VA_ARGS__

EVAL(WHILE(PRED,OP,CONT,MAX_DIMENSION))


