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

#include "loader.hpp"

#include <iostream>
#include <dlfcn.h>

using namespace Manicore;

template<size_t dimension>
Maps_loader<dimension>::Maps_loader(const char *filename) {
  _handle = dlopen(filename, RTLD_LAZY);
  if (_handle == nullptr) {
    std::cerr<<"Failed to open map list: "<<dlerror()<<std::endl;
    throw std::runtime_error("Failed to open shared library");
  }
  _setup_maps();
}

template<>
void Maps_loader<2>::_setup_maps() {
  char *error;
  _3D_embedding = reinterpret_cast<decltype(_3D_embedding)>(dlsym(_handle,"List_embedding_2to3"));
  if((error = dlerror()) != NULL) {
    _3D_embedding = DefaultMapping<3,2>;
  }
  _metrics = reinterpret_cast<decltype(_metrics)>(dlsym(_handle,"List_metrics_2D"));
  if((error = dlerror()) != NULL) {
    std::cerr<<"Failed to retrieve edge map list: "<<error<<std::endl;
    throw std::runtime_error("Failed to retrieve edge maps");
  }
  _edge_maps = reinterpret_cast<decltype(_edge_maps)>(dlsym(_handle,"List_edge_maps_2D"));
  if((error = dlerror()) != NULL) {
    std::cerr<<"Failed to retrieve edge map list: "<<error<<std::endl;
    throw std::runtime_error("Failed to retrieve edge maps");
  }
  _face_maps = reinterpret_cast<decltype(_face_maps)>(dlsym(_handle,"List_face_maps_2D"));
  if((error = dlerror()) != NULL) {
    std::cerr<<"Failed to retrieve face map list: "<<error<<std::endl;
    throw std::runtime_error("Failed to retrieve face maps");
  }
  _edge_pullbacks = reinterpret_cast<decltype(_edge_pullbacks)>(dlsym(_handle,"List_edge_pullbacks_2D"));
  if((error = dlerror()) != NULL) {
    std::cerr<<"Failed to retrieve edge map list: "<<error<<std::endl;
    throw std::runtime_error("Failed to retrieve edge pullbacks");
  }
  _face_pullbacks = reinterpret_cast<decltype(_face_pullbacks)>(dlsym(_handle,"List_face_pullbacks_2D"));
  if((error = dlerror()) != NULL) {
    std::cerr<<"Failed to retrieve face map list: "<<error<<std::endl;
    throw std::runtime_error("Failed to retrieve face pullbacks");
  }
}

template<size_t dimension>
Maps_loader<dimension>::~Maps_loader() {
  dlclose(_handle);
}


template<size_t dimension>
ParametrizedMap<3,dimension>* Maps_loader<dimension>::get_new_embedding_3D(size_t id) const {
  auto func = (*_3D_embedding)(id);
  return func;
}
template<size_t dimension>
ParametrizedMap<3,dimension>* Maps_loader<dimension>::get_new_embedding_3D(size_t id, std::vector<double> const &extra) const {
  auto func = (*_3D_embedding)(id);
  if (extra.size() > 0) {
    func->_extra = extra;
  }
  return func;
}
template<size_t dimension>
ParametrizedMetricMap<dimension>* Maps_loader<dimension>::get_new_metrics(size_t id) const {
  auto func = (*_metrics)(id);
  return func;
}
template<size_t dimension>
ParametrizedMetricMap<dimension>* Maps_loader<dimension>::get_new_metrics(size_t id, std::vector<double> const &extra) const {
  auto func = (*_metrics)(id);
  if (extra.size() > 0) {
    func->_extra = extra;
  }
  return func;
}
// Elements maps
template<size_t dimension>
ParametrizedMap<dimension,1>* Maps_loader<dimension>::get_new_edge_map(size_t id) const {
  auto func = (*_edge_maps)(id);
  return func;
}
template<size_t dimension>
ParametrizedMap<dimension,1>* Maps_loader<dimension>::get_new_edge_map(size_t id, std::vector<double> const &extra) const {
  auto func = (*_edge_maps)(id);
  if (extra.size() > 0) {
    func->_extra = extra;
  }
  return func;
}
template<size_t dimension>
ParametrizedMap<dimension,2>* Maps_loader<dimension>::get_new_face_map(size_t id) const {
  auto func = (*_face_maps)(id);
  return func;
}
template<size_t dimension>
ParametrizedMap<dimension,2>* Maps_loader<dimension>::get_new_face_map(size_t id, std::vector<double> const &extra) const {
  auto func = (*_face_maps)(id);
  if (extra.size() > 0) {
    func->_extra = extra;
  }
  return func;
}

template<size_t dimension>
ParametrizedMap<dimension,3>* Maps_loader<dimension>::get_new_cell_map(size_t id) const {
  auto func = (*_cell_maps)(id);
  return func;
}
template<size_t dimension>
ParametrizedMap<dimension,3>* Maps_loader<dimension>::get_new_cell_map(size_t id, std::vector<double> const &extra) const {
  auto func = (*_cell_maps)(id);
  if (extra.size() > 0) {
    func->_extra = extra;
  }
  return func;
}
// Pullbacks
template<size_t dimension>
ParametrizedDerivedMap<dimension,1>* Maps_loader<dimension>::get_new_edge_pullbacks(size_t id) const {
  auto func = (*_edge_pullbacks)(id);
  return func;
}
template<size_t dimension>
ParametrizedDerivedMap<dimension,1>* Maps_loader<dimension>::get_new_edge_pullbacks(size_t id, std::vector<double> const &extra) const {
  auto func = (*_edge_pullbacks)(id);
  if (extra.size() > 0) {
    func->_extra = extra;
  }
  return func;
}
template<size_t dimension>
ParametrizedDerivedMap<dimension,2>* Maps_loader<dimension>::get_new_face_pullbacks(size_t id) const {
  auto func = (*_face_pullbacks)(id);
  return func;
}
template<size_t dimension>
ParametrizedDerivedMap<dimension,2>* Maps_loader<dimension>::get_new_face_pullbacks(size_t id, std::vector<double> const &extra) const {
  auto func = (*_face_pullbacks)(id);
  if (extra.size() > 0) {
    func->_extra = extra;
  }
  return func;
}

template<size_t dimension>
ParametrizedDerivedMap<dimension,3>* Maps_loader<dimension>::get_new_cell_pullbacks(size_t id) const {
  auto func = (*_cell_pullbacks)(id);
  return func;
}
template<size_t dimension>
ParametrizedDerivedMap<dimension,3>* Maps_loader<dimension>::get_new_cell_pullbacks(size_t id, std::vector<double> const &extra) const {
  auto func = (*_cell_pullbacks)(id);
  if (extra.size() > 0) {
    func->_extra = extra;
  }
  return func;
}

#include "preprocessor.hpp"
#define PRED(x, ...) COMPL(IS_1(x))
#define OP(x, ...) template class Manicore::Maps_loader<x>;
#define CONT(x, ...) DEC(x), __VA_ARGS__

EVAL(WHILE(PRED,OP,CONT,MAX_DIMENSION))


