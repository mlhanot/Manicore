#include <iostream>

#include "exterior_objects.hpp"

using namespace Manicore;

constexpr int max_degree = 3;
constexpr size_t max_dimension = 5;


int main() {

  // Initialize the module
  Initialize_exterior_module<max_dimension>::init(max_degree);
  auto print = []<size_t _l,size_t _d>(auto && print) {
    std::cout<<"Hodge star in dimension "<<_d<<" for "<<_l<<"-forms\n"<<ComplBasis<_l,_d>::compute()<<std::endl;
    if constexpr (_l < _d) print.template operator()<_l+1,_d>(print);
    else if constexpr (_d < max_dimension) print.template operator()<0,_d+1>(print);
  };
  print.template operator()<0,1>(print);

  return 0;
}

