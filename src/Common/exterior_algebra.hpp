// Core data structure for spaces and operators on manifolds.
//
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

#ifndef EXTERIOR_ALGEBRA_HPP
#define EXTERIOR_ALGEBRA_HPP

#include "exterior_dimension.hpp"

// Std utilities
#include <vector>
#include <array>
#include <unordered_map>
#include <algorithm> 
#include <cstdlib>
#include <cassert>

// Linear algebra utilities
#include <Eigen/Dense> 
#include <unsupported/Eigen/KroneckerProduct> // Used to couple the action on polynomial and on the exterior algebra

namespace Manicore {
   /** \file exterior_algebra.hpp
     The methods in this file are meant to compute the action of everything that is independent of the atlas.

     The most useful are:
      Compute_pullback : computes the action of a pullback to the exterior algebra
      Monomial_full : gives a mapping between index and monomial powers

     The polynomial basis must be initialized before use.

     The explicit ordering for low dimensions is available [here](./ref_exterior_algebra).
   */
  

  ///------------------------------------------------------------------------------------------------------------------------------
  // Exterior algebra
  ///------------------------------------------------------------------------------------------------------------------------------
  /* Contains the part of methods dealing with the exterior algebra basis
    */

  /// \addtogroup ExteriorBundle
  ///@{

  /// Class to handle the exterior algebra basis
  template<size_t l, size_t d>
  class ExteriorBasis 
  {
    public:
      /// Initialize the exterior algebra basis. Must be called at least once before use
      static void init() noexcept {
        static_assert(0 <= l && l <= d,"Error: Tried to generate the exterior algebra basis outside the range [0,d]");
        if (initialized == 1) return; // initialize only the first time
        initialized = 1;
        size_t acc = 0;
        std::array<size_t, l> cbasis;
        for (size_t i = 0; i < l; ++i) {
          cbasis[i] = i; // fill to first element
        }
        _basis.try_emplace(acc,cbasis);
        while (_find_next_tuple(cbasis)) {
          ++acc;
          _basis.try_emplace(acc,cbasis);
        };
      };

      /// Return the coefficient of the basis at the given index
      static const std::array<size_t,l> & expand_basis(size_t i) 
      {
        assert(initialized==1 && "ExteriorBasis was not initialized");
        return _basis.at(i);
      }

      /// Search the index corresponding to the given tuple. Throw if not found
      static const size_t index_from_tuple(const std::array<size_t,l> & tuple /*!< Always ordered, e.g. (0,2,3)*/) {
        auto found = std::find_if(_basis.begin(), _basis.end(), [&tuple](const auto& p)
                                                                {return p.second == tuple;});
        if (found != _basis.end()) {
          return found->first;
        } else {
          throw; // tuple not in range
        }
      }
    
    private:
      // Unordered_map provide faster lookup time than map
      static inline std::unordered_map<size_t, std::array<size_t, l>> _basis;
      static inline int initialized = 0; // We need inline to be able to define it inside the class
      // Helper, find the next element of the basis, return false if this is the last element
      static bool _find_next_tuple(std::array<size_t, l> &basis) noexcept {
        for (size_t i = 0; i < l; ++i) {
          size_t nval = basis[l - i - 1] + 1;
          if (nval < d - i) { // value valid, we must reset everything after
            basis[l - i - 1] = nval;
            for (size_t j = 1; j <= i ; ++j){
              basis[l - i - 1 + j] = nval + j;
            }
            return true;
          } else { // already using the last element available
            continue;
          }
        }
        return false;
      };
  };

  /// Return a mapping from the basis of l-forms in dimension d to the basis of (d-l)-forms 
  template<size_t l, size_t d>
  class ComplBasis {
    public:
      static Eigen::Matrix<double,Dimension::ExtDim(d-l,d),Dimension::ExtDim(l,d)> const & compute() {
        static Eigen::Matrix<double,Dimension::ExtDim(d-l,d),Dimension::ExtDim(l,d)> mat = _compute();
        return mat;
      }
    private:
      static int _get_basis_sign(std::array<size_t,l> const &ori_basis, std::array<size_t,d-l> const &star_basis) {
        size_t ori_rem = l;
        size_t star_rem = d-l;
        int sign = 1;
        // Parse both basis to try to reconstitute the volume form, return the sign of the transformation, or 0 if this fail
        for (size_t c_t = 0; c_t < d; ++c_t) {
          if (ori_rem > 0 && ori_basis[l-ori_rem] == c_t) {
            ori_rem--;
            if (star_rem > 0 && star_basis[d-l-star_rem] == c_t) {
              return 0; // Element present in both basis
            } else {
              continue;
            }
          } else {
            if (star_rem > 0 && star_basis[d-l-star_rem] == c_t) {
              star_rem--;
              sign *= (ori_rem%2 == 0)? 1 : -1;
              continue;
            } else {
              return 0; // Element not present in any basis
            }
          }
        }
        return sign;
      }
      static Eigen::Matrix<double,Dimension::ExtDim(d-l,d),Dimension::ExtDim(l,d)> _compute() {
        if constexpr (l == 0 || l == d) {
          return Eigen::Matrix<double,1,1>{1.};
        } else {
          Eigen::Matrix<double,Dimension::ExtDim(d-l,d),Dimension::ExtDim(l,d)> rv;
          for (size_t i_r = 0; i_r < Dimension::ExtDim(l,d); ++i_r) {
            std::array<size_t,l> const & ori_basis = ExteriorBasis<l,d>::expand_basis(i_r);
            for (size_t i_l = 0; i_l < Dimension::ExtDim(d-l,d); ++i_l) {
              std::array<size_t,d-l> const & star_basis = ExteriorBasis<d-l,d>::expand_basis(i_l);
              rv(i_l,i_r) = _get_basis_sign(ori_basis,star_basis);
            }
          }
          return rv;
        }
      }
  };

  ///------------------------------------------------------------------------------------------------------------------------------
  // Pullback helpers
  ///------------------------------------------------------------------------------------------------------------------------------

  /// Generic determinant computation.
  /** The first two arguments should be the list of indexes to use, and the last the matrix
    This function returns the determinant of the partial matrix
    */
  template<typename V, typename Derived>
  double Compute_partial_det(const V& a1, const V& a2, const Eigen::MatrixBase<Derived>& A) {
    constexpr size_t N = std::tuple_size<V>::value;
    if constexpr (N==0) {
      return 0.;
    } else if constexpr (N == 1) {
      return A(a1[0],a2[0]);
    } else {
      double sign = 1.;
      double sum = 0.;
      std::array<typename V::value_type,N-1> b1,b2;
      std::copy(a1.cbegin()+1,a1.cend(),b1.begin());
      std::copy(a2.cbegin()+1,a2.cend(),b2.begin());
      for (size_t i = 0; i < N-1;++i){
        sum += sign*A(a1[i],a2[0])*Compute_partial_det(b1,b2,A);
        sign *= -1.;
        b1[i] = a1[i];
      }
      sum += sign*A(a1[N-1],a2[0])*Compute_partial_det(b1,b2,A);
      return sum;
    }
  }

  /// Generic pullback computation.
  /** The matrix A go from the space 1 to the space 2
    Specialized for some case, try to avoid the generic definition
  */
  template<size_t l, size_t d1, size_t d2>
  struct Compute_pullback {
    template<typename Derived> 
    static Eigen::Matrix<double, Dimension::ExtDim(l,d1), Dimension::ExtDim(l,d2)> compute(Eigen::MatrixBase<Derived> const & A) {
      Eigen::Matrix<double, Dimension::ExtDim(l,d1), Dimension::ExtDim(l,d2)> rv;
      for (size_t i = 0; i < Dimension::ExtDim(l,d1); ++i) {
        for (size_t j = 0; j < Dimension::ExtDim(l,d2); ++j) {
          rv(i,j) = Compute_partial_det(ExteriorBasis<l,d2>::expand_basis(j),ExteriorBasis<l,d1>::expand_basis(i),A);
        }
      }
      return rv;
    }
  };

  template<size_t d1, size_t d2> 
  struct Compute_pullback<0,d1,d2> {
    template<typename Derived> 
    static Eigen::Matrix<double,1,1> compute(Eigen::MatrixBase<Derived> const & A) {
      return Eigen::Matrix<double,1,1>{1.};
    }
  };

  template<size_t d1, size_t d2> 
  struct Compute_pullback<1,d1,d2> {
    template<typename Derived> 
    static Eigen::Matrix<double,d1,d2> compute(Eigen::MatrixBase<Derived> const & A) {
      return A.transpose();
    }
  };

  template<size_t d> 
  struct Compute_pullback<d,d,d> {
    template<typename Derived> 
    static Eigen::Matrix<double,1,1> compute(Eigen::MatrixBase<Derived> const & A) {
      return Eigen::Matrix<double,1,1>{A.determinant()};
    }
  };

  template<> 
  struct Compute_pullback<1,1,1> {
    template<typename Derived> 
    static Eigen::Matrix<double,1,1> compute(Eigen::MatrixBase<Derived> const & A) {
      return Eigen::Matrix<double,1,1>{A(0,0)};
    }
  };

  template<>
  struct Compute_pullback<2,2,3> {
    template<typename Derived> 
    static Eigen::Matrix<double,1,3> compute(Eigen::MatrixBase<Derived> const & A) {
      return Eigen::Matrix<double,1,3>{{A(0,0)*A(1,1) - A(0,1)*A(1,0), A(0,0)*A(2,1) - A(0,1)*A(2,0), A(1,0)*A(2,1) - A(1,1)*A(2,0)}};
    }
  };
  template<>
  struct Compute_pullback<2,3,2> {
    template<typename Derived> 
    static Eigen::Matrix<double,3,1> compute(Eigen::MatrixBase<Derived> const & A) {
      return Eigen::Matrix<double,3,1>{A(0,0)*A(1,1) - A(0,1)*A(1,0), A(0,0)*A(1,2) - A(0,2)*A(1,0), A(0,1)*A(1,2) - A(0,2)*A(1,1)};
    }
  };
  template<>
  struct Compute_pullback<2,3,3> {
    template<typename Derived> 
    static Eigen::Matrix<double,3,3> compute(Eigen::MatrixBase<Derived> const & A) {
      return Eigen::Matrix<double,3,3>{
        {A(0,0)*A(1,1) - A(0,1)*A(1,0), A(0,0)*A(2,1) - A(0,1)*A(2,0), A(1,0)*A(2,1) - A(1,1)*A(2,0)},
        {A(0,0)*A(1,2) - A(0,2)*A(1,0), A(0,0)*A(2,2) - A(0,2)*A(2,0), A(1,0)*A(2,2) - A(1,2)*A(2,0)},
        {A(0,1)*A(1,2) - A(0,2)*A(1,1), A(0,1)*A(2,2) - A(0,2)*A(2,1), A(1,1)*A(2,2) - A(1,2)*A(2,1)}
      };
    }
  };

  /// Wrapper for the \f$L^2\f$ product on the exterior algebra.
  /** The matrix g is the metric on the cotangent space. */
  template<size_t l>
  struct Compute_ExtGram {
    template<typename Derived> 
    static Eigen::Matrix<double, Dimension::ExtDim(l,Eigen::internal::traits< Derived >::RowsAtCompileTime), 
                                 Dimension::ExtDim(l,Eigen::internal::traits< Derived >::RowsAtCompileTime)> 
    compute(Eigen::MatrixBase<Derived> const & g) {
      constexpr auto N = Eigen::internal::traits< Derived >::RowsAtCompileTime;
      static_assert(N == Eigen::internal::traits< Derived >::ColsAtCompileTime);
      static_assert(N > 0);
      return Compute_pullback<l,N,N>::compute(g);
    }
  };


  ///------------------------------------------------------------------------------------------------------------------------------
  // Polynomial algebra
  ///------------------------------------------------------------------------------------------------------------------------------
  /* Contains the part of methods dealing with the polynomial basis
    */

  /// Generate a basis of monomial powers of degree r
  template<size_t d> 
  struct Monomial_powers {
    /// Basis of homogeneous polynomial of degree r
    static std::vector<std::array<size_t, d>> const & homogeneous(const int r) {
      assert(r <= _init_deg && "Error: Monomial_powers must be initialized first");
      return _powers[r];
    }

    /// Basis of polynomial of at most degree r
    static std::vector<std::array<size_t,d>> const & complete(const int r) {
      assert(r <= _init_deg && "Error: Monomial_powers must be initialized first");
      return _powers_full[r];
    }

    /// Initialise the basis for degree up to r. Must be called at least once before use
    static void init(const int r) {
      static_assert(d > 0,"Error: Tried to construct a monomial basis in dimension 0");
      assert(r <= _max_deg && "Maximum degree reached, increase _max_deg if needed");
      if (r >_init_deg) {
        for (int h_r = _init_deg + 1; h_r <= r; ++h_r) { // init homogeneous
          std::vector<std::array<size_t,d>> powers;
          std::array<size_t,d> current;
          inner_loop(0,h_r,current,powers);
          _powers[h_r] = powers;
          if (h_r == 0) {
            _powers_full[0] = powers;
          } else {
            _powers_full[h_r] = _powers_full[h_r-1];
            for (size_t j = 0; j < powers.size(); ++j) {
              _powers_full[h_r].emplace_back(powers[j]);
            }
          }
        }
        _init_deg = r;
      }
    }

    private:
      static constexpr int _max_deg = 20;
      static inline int _init_deg = -1;
      static inline std::array<std::vector<std::array<size_t,d>>,_max_deg+1> _powers;
      static inline std::array<std::vector<std::array<size_t,d>>,_max_deg+1> _powers_full;

      static void inner_loop(size_t cindex, int max, std::array<size_t,d> & current, std::vector<std::array<size_t,d>> & powers) {
        if (cindex == d-1) {
          current[cindex] = max; // Last direction to enforce the degree
          powers.emplace_back(current);
        } else {
          for(int i = 0; i <= max; ++i) {
            current[cindex] = i;
            inner_loop(cindex+1,max-i,current,powers);
          }
        }
      }
  };
  ///@}

} // End namespace

#endif
