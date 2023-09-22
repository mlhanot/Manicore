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

#include "exporter.hpp"

#include "integral.hpp"

#include <fstream>
#include <iostream>

using namespace Manicore;

template<size_t dimension>
Exporter<dimension>::Exporter(Mesh<dimension> const * mesh,int r,int acc)
{
  Integral<dimension,dimension> integral(mesh);
  _locations.reserve(mesh->n_cells(dimension));
  _cellIndices.reserve(mesh->n_cells(dimension));
  _metricInv.reserve(mesh->n_cells(dimension));
  _volume.reserve(mesh->n_cells(dimension));
  _polyEval.reserve(mesh->n_cells(dimension));
  _pushforward.reserve(mesh->n_cells(dimension));
  if constexpr (dimension > 2) _pushforwardStar.reserve(mesh->n_cells(dimension));
  for (size_t iT = 0; iT < mesh->n_cells(dimension); ++iT) {
    QuadratureRule<dimension> quad = integral.generate_quad(iT,acc);
    for (auto quadv : quad) {
      auto const & T = mesh->template get_cell_map<dimension>(iT);
      Eigen::VectorXd chartX = T.evaluate_I(0,quadv.vector);
      size_t mId = mesh->get_map_ids(dimension,iT)[0];
      // Physical location on the embedded manifold
      _locations.push_back(mesh->get_3D_embedding(mId,chartX)); 
      _cellIndices.push_back(iT);
      // Metric at the location on the reference
      {
        auto const DJ = T.evaluate_DJ(0,chartX);
        _metricInv.push_back(DJ*mesh->metric_inv(mId,chartX)*DJ.transpose()); 
      }
      _volume.push_back(T.template evaluate_DI_p<dimension>(0,quadv.vector)[0]*
              mesh->volume_form(mId,chartX)*mesh->orientationTopCell(iT));
      // Precompute the polynomial evaluations
      size_t dimP = Dimension::PolyDim(r,dimension);
      Eigen::VectorXd polyEval(dimP);
      for (size_t iP = 0; iP < dimP; ++iP) {
        polyEval(iP) = T.evaluate_poly_on_ref(quadv.vector,iP,r);
      }
      _polyEval.push_back(polyEval);
      // Store the pushforward from the reference to the chart
      _pushforward.push_back(mesh->get_3D_pushforward(mId,chartX)
                            *mesh->metric_inv(mId,chartX) 
                            *T.template evaluate_DJ_p<1>(0,chartX));
      _pushforwardStar.push_back(_pushforward.back()*mesh->template getHodge<dimension-1>(iT,quadv.vector));
    } // end quadv
  } // end iT
}

template<size_t dimension>
int Exporter<dimension>::save(size_t k, std::function<Eigen::VectorXd(size_t iT)> Fu_h, const char *filename, bool star) const 
{
  std::fstream fh{filename, fh.trunc | fh.out};
  if (!fh.is_open()){
    std::cerr<<"Failed to open "<<filename<<std::endl;
    return 1;
  }
  if (Dimension::ExtDim(k,dimension) == 1) {
    fh << "X,Y,Z,Val\n";
  } else if (Dimension::ExtDim(k,dimension) == dimension) {
    fh << "X,Y,Z,ValX,ValY,ValZ\n";
  }
  size_t oldIT = -1;
  Eigen::VectorXd uT;
  for (size_t iLoc = 0; iLoc < _locations.size(); ++iLoc) {
    if (_cellIndices[iLoc] != oldIT) { // Prevent several consecutive restrictions
      oldIT = _cellIndices[iLoc];
      uT = Fu_h(oldIT);
    }
    auto const & x = _locations[iLoc];
    if (Dimension::ExtDim(k,dimension) == 1) { // 0 or n forms
      double evalP = _polyEval[iLoc].dot(uT); // Dim 1, hence scalar
      if (k == dimension) evalP /= _volume[iLoc]; // As factor of the volume form
      fh << x(0) <<"," << x(1) << "," << x(2) << "," << evalP << "\n";
    } else if (Dimension::ExtDim(k,dimension) == dimension) { // 1 or n-1 forms
      Eigen::Vector<double,dimension> evalP = 
        Eigen::KroneckerProduct(Eigen::Matrix<double,dimension,dimension>::Identity(), _polyEval[iLoc].transpose())*uT;
      Eigen::Vector3d embP = (k == 1)? 
        (star? _pushforwardStar[iLoc]*evalP : _pushforward[iLoc]*evalP) : 
        _pushforwardStar[iLoc]*evalP;
      fh << x(0) <<"," << x(1) << "," << x(2) << ",";
      fh << embP(0) << "," << embP(1) << "," << embP(2) << "\n";
    } else {
      std::cerr<<"Writer not yet implemented for other form degree"<<std::endl;
      return 2;
    }
  } // for iLoc
  return 0;
}

template<size_t dimension>
int Exporter<dimension>::saveSq(size_t k, std::function<Eigen::VectorXd(size_t iT)> Fu_h, const char *filename) const 
{
  std::fstream fh{filename, fh.trunc | fh.out};
  if (!fh.is_open()){
    std::cerr<<"Failed to open "<<filename<<std::endl;
    return 1;
  }
  fh << "X,Y,Z,Val\n";
  size_t oldIT = -1;
  Eigen::VectorXd uT;
  for (size_t iLoc = 0; iLoc < _locations.size(); ++iLoc) {
    if (_cellIndices[iLoc] != oldIT) { // Prevent several consecutive restrictions
      oldIT = _cellIndices[iLoc];
      uT = Fu_h(oldIT);
    }
    auto const & x = _locations[iLoc];
    fh << x(0) <<"," << x(1) << "," << x(2) << ",";
    if (Dimension::ExtDim(k,dimension) == 1) { // 0 or n forms
      double evalP = _polyEval[iLoc].dot(uT); // Dim 1, hence scalar
      if (k == dimension) evalP /= _volume[iLoc]; // As factor of the volume form
      fh << evalP*evalP << "\n";
    } else if (k == 1) {
      Eigen::Vector<double,dimension> evalP = 
        Eigen::KroneckerProduct(Eigen::Matrix<double,dimension,dimension>::Identity(), _polyEval[iLoc].transpose())*uT;
      double nVal = (evalP.transpose()*_metricInv[iLoc]*evalP)(0);
      fh << nVal << "\n";
    } else {
      std::cerr<<"Writer not yet implemented for other form degree"<<std::endl;
      return 2;
    }
  } // for iLoc
  return 0;
}

#include "preprocessor.hpp"
// Instantiate the class for all dimensions
#define PRED(x, ...) COMPL(IS_1(x))
#define OP(x, ...) template class Manicore::Exporter<x>;
#define CONT(x, ...) DEC(x), __VA_ARGS__
EVAL(WHILE(PRED,OP,CONT,MAX_DIMENSION))
