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

#ifndef EXTERIOR_OBJECT_HPP
#define EXTERIOR_OBJECT_HPP

#include "exterior_algebra.hpp"

// Linear algebra utilities
#include <unsupported/Eigen/KroneckerProduct> // Used to couple the action on polynomial and on the exterior algebra

namespace Manicore {
   /** \file exterior_objects.hpp
     Compute the action of Kozsul and Diff on the exterior algebra.
   The action is returned as a list of matrix between the exterior algebra basis
   To get the full action, one must take the Kronecker product between the action on the exterior algebra and the action on the polynomial space.
   In the case of Koszul, the action is to multiply by \f$x_i\f$, and in the case of Diff, it is the differentiated by \f$x_i\f$. 

   The basis are those given in exterior_algebra.hpp.
   The basis of \f$\mathcal{P}_r\Lambda^l(\mathbb{R}^d)\f$ is the Kronecker product \f$\Lambda^l \otimes \mathcal{P}_r\f$.
  

     The most useful are:
      Koszul_full : gives the matrix of the Koszul operator
      Diff_full : gives the matrix of the Diff operator

     Call Manicore::Initialize_exterior_module<d>::init(int r) to initialize every module on dimension d
*/

  ///------------------------------------------------------------------------------------------------------------------------------
  // Exterior algebra
  ///------------------------------------------------------------------------------------------------------------------------------

  /// Koszul operator on the exterior algebra
  template<size_t l, size_t d>
  class Koszul_exterior {
    public:
      typedef Eigen::Matrix<double, Dimension::ExtDim(l-1,d), Dimension::ExtDim(l,d)> ExtAlgMatType;

      /// Return the action of the Koszul operator on the i-th basis of the exterior algebra
      static const ExtAlgMatType & get_transform(size_t i) {
        return _transforms.at(i);
      }

      /// Initialize the Koszul operator
      static void init() noexcept {
        static_assert(0 < l && l <= d,"Error: Tried to generate Koszul basis outside the range [1,d]");
        if (initialized == 1) return;
        initialized = 1;
        ExteriorBasis<l-1,d>::init(); // ensure that the exterior basis is initialized
        ExteriorBasis<l,d>::init(); // ensure that the exterior basis is initialized
        for (size_t i = 0; i < d; ++i) {
          _transforms[i].setZero();
        }
        if (l == 1) { // special case, ext_basis = \emptyset
          for (size_t i = 0; i < d; ++i) {
            _transforms[i](0,i) = 1;
          }
          return;
        } 
        for (size_t i = 0; i < Dimension::ExtDim(l-1,d); ++i) {
          int sign = 1;
          size_t try_pos = 0;
          const std::array<size_t, l-1> & cbasis = ExteriorBasis<l-1,d>::expand_basis(i);
          std::array<size_t,l> basis_cp;
          std::copy(cbasis.cbegin(),cbasis.cend(),basis_cp.begin()+1);
          // basis_cp contains a copy of cbasis with one more slot
          for (size_t j = 0; j < d; ++j) {
            if (try_pos == l-1 || j < cbasis[try_pos]) { // insert here
              basis_cp[try_pos] = j;
              _transforms[j](i,ExteriorBasis<l,d>::index_from_tuple(basis_cp)) = sign;
            } else { // j == cbasis[try_pos]
              basis_cp[try_pos] = basis_cp[try_pos+1]; 
              ++try_pos;
              sign = -sign;
              continue;
            }
          }
        }
      }; // end init()

    private:
      static inline int initialized = 0;
      static inline std::array<ExtAlgMatType,d> _transforms;
  };

  /// Diff operator on the exterior algebra
  template<size_t l, size_t d>
  class Diff_exterior {
    public:
      typedef Eigen::Matrix<double, Dimension::ExtDim(l+1,d), Dimension::ExtDim(l,d)> ExtAlgMatType;

      /// Return the action of the Diff operator on the i-th basis of the exterior algebra
      static const ExtAlgMatType & get_transform(size_t i) {
        return _transforms.at(i);
      }

      /// Initialize the Diff operator
      static void init() noexcept {
        static_assert(l < d,"Error: Tried to generate diff basis outside the range [0,d-1]");
        if (initialized == 1) return;
        initialized = 1;
        Koszul_exterior<l+1,d>::init();
        for (size_t i = 0; i < d; ++i) {
          _transforms[i] = Koszul_exterior<l+1,d>::get_transform(i).transpose().eval();
        }
      };
    private:
      static inline int initialized = 0;
      static inline std::array<ExtAlgMatType,d> _transforms;
  };

  ///------------------------------------------------------------------------------------------------------------------------------
  // Polynomial algebra
  ///------------------------------------------------------------------------------------------------------------------------------

  /// Generate the matrices for the Koszul operator on homogeneous monomial
  template<size_t d, size_t index>
  struct Koszul_homogeneous_mat {
    static Eigen::MatrixXd get (const int r) {
      static_assert(index < d,"Error: Tried to take the koszul operator on a direction outside the dimension");
      Eigen::MatrixXd M = Eigen::MatrixXd::Zero(Dimension::HDim(r+1,d), Dimension::HDim(r,d));
      for (size_t i = 0; i < Dimension::HDim(r,d); ++i) {
        std::array<size_t, d> current = Monomial_powers<d>::homogeneous(r)[i];
        current.at(index) += 1;
        size_t val = std::find(Monomial_powers<d>::homogeneous(r+1).cbegin(), 
                               Monomial_powers<d>::homogeneous(r+1).cend(),current) 
                    - Monomial_powers<d>::homogeneous(r+1).cbegin();
        M(val,i) = 1.;
      }
      return M;
    }
  };

  /// Generate the matrices for the Differential operator on homogeneous monomial
  template<size_t d, size_t index>
  struct Diff_homogeneous_mat {
    static Eigen::MatrixXd get (const int r) {
      static_assert(index < d,"Error: Tried to take the differential operator on a direction outside the dimension");
      assert(r > 0 && "Error: Cannot generate a matrix for the differential on P_0");
      Eigen::MatrixXd M = Eigen::MatrixXd::Zero(Dimension::HDim(r-1,d), Dimension::HDim(r,d)); 
      for (size_t i = 0; i < Dimension::HDim(r,d); ++i) {
        std::array<size_t, d> current = Monomial_powers<d>::homogeneous(r)[i];
        size_t cval = current.at(index);
        if (cval == 0) continue; // Zero on that col
        current.at(index) -= 1;
        size_t val = std::find(Monomial_powers<d>::homogeneous(r-1).cbegin(), 
                               Monomial_powers<d>::homogeneous(r-1).cend(),current) 
                    - Monomial_powers<d>::homogeneous(r-1).cbegin();
        M(val,i) = cval;
      }
      return M;
    }
  };

  ///------------------------------------------------------------------------------------------------------------------------------
  // Coupling polynomial and exterior
  ///------------------------------------------------------------------------------------------------------------------------------
  /* Couple the part on the exterior algebra and the part on polynomials
    */

  /// \addtogroup ExteriorBundle
  ///@{

  /// Koszul operator from \f$\mathcal{P}_r\Lambda^l(\mathbb{R}^d)\f$ to \f$\mathcal{P}_{r+1}\Lambda^{l-1}(\mathbb{R}^d)\f$.
  template<size_t l, size_t d>
  struct Koszul_full {
  /// Koszul operator from \f$\mathcal{P}_r\Lambda^l(\mathbb{R}^d)\f$ to \f$\mathcal{P}_{r+1}\Lambda^{l-1}(\mathbb{R}^d)\f$.
    static Eigen::MatrixXd get (const int r /*!< Polynomial degree */) {
      if constexpr (l==0 || l > d) {
        return Eigen::MatrixXd(0,0);
      }
      Eigen::MatrixXd M(Dimension::PLDim(r+1,l-1,d),Dimension::PLDim(r,l,d));
      M.setZero();
      if (Dimension::PLDim(r+1,l-1,d) > 0 && Dimension::PLDim(r,l,d) > 0) {
        Eigen::MatrixXd scalar_part(Dimension::PolyDim(r+1,d),Dimension::PolyDim(r,d));
        init_loop_for<0>(r,scalar_part,M);
      }
      return M;
    }

    private:
      template<size_t i, typename T> static void init_loop_for(int r, T & scalar_part,T & M) {
        if constexpr(i < d) {
          scalar_part.setZero();
          int offset_l = Dimension::HDim(0,d);
          int offset_c = 0;
          for (int s = 0; s <= r; ++s) {
            int inc_l = Dimension::HDim(s+1,d);
            int inc_c = Dimension::HDim(s,d);
            scalar_part.block(offset_l,offset_c,inc_l,inc_c) = Koszul_homogeneous_mat<d,i>::get(s);
            offset_l += inc_l;
            offset_c += inc_c;
          }
          M += Eigen::KroneckerProduct(Koszul_exterior<l,d>::get_transform(i),scalar_part);
          init_loop_for<i+1>(r,scalar_part,M);
        }
      }
  };

  /// Differential operator from \f$\mathcal{P}_r\Lambda^l(\mathbb{R}^d)\f$ to \f$\mathcal{P}_{r-1}\Lambda^{l+1}(\mathbb{R}^d)\f$.
  template<size_t l, size_t d>
  struct Diff_full {
  /// Differential operator from \f$\mathcal{P}_r\Lambda^l(\mathbb{R}^d)\f$ to \f$\mathcal{P}_{r-1}\Lambda^{l+1}(\mathbb{R}^d)\f$.
    static Eigen::MatrixXd get (const int r /*!< Polynomial degree */) {
      if (l >= d) {
        return Eigen::MatrixXd(0,0);
      }
      Eigen::MatrixXd M(Dimension::PLDim(r-1,l+1,d),Dimension::PLDim(r,l,d));
      M.setZero();
      if (Dimension::PLDim(r-1,l+1,d) > 0 || Dimension::PLDim(r,l,d) > 0) {
        Eigen::MatrixXd scalar_part(Dimension::PolyDim(r-1,d),Dimension::PolyDim(r,d));
        init_loop_for<0>(r,scalar_part,M);
      }
      return M;
    };

  /// Differential operator from \f$\mathcal{P}_r\Lambda^l(\mathbb{R}^d)\f$ to \f$\mathcal{P}_{r}\Lambda^{l+1}(\mathbb{R}^d)\f$.
    static Eigen::MatrixXd get_as_degr (const int r /*!< Polynomial degree */) {
      auto inj = Eigen::KroneckerProduct(Eigen::Matrix<double,Dimension::ExtDim(l+1,d),Dimension::ExtDim(l+1,d)>::Identity(),
                                 Eigen::MatrixXd::Identity(Dimension::PolyDim(r,d),Dimension::PolyDim(r-1,d)));
      return inj*get(r);
    };

    private:
      template<size_t i,typename T> static void init_loop_for(int r,T &scalar_part,T &M) {
        if constexpr (i < d) {
          scalar_part.setZero();
          int offset_l = 0;
          int offset_c = Dimension::HDim(0,d);
          for (int s = 0; s < r; ++s) {
            int inc_l = Dimension::HDim(s,d);
            int inc_c = Dimension::HDim(s+1,d);
            scalar_part.block(offset_l,offset_c,inc_l,inc_c) = Diff_homogeneous_mat<d,i>::get(s+1);
            offset_l += inc_l;
            offset_c += inc_c;
          }
          M += Eigen::KroneckerProduct(Diff_exterior<l,d>::get_transform(i),scalar_part);
          init_loop_for<i+1>(r,scalar_part,M);
        }
      }
  };

  /// Initialize every class related to the polynomial degree r
  template<size_t d>
  struct Initialize_exterior_module{
    /// Initialize up to degree r
    static void init(int r /*!< Polynomial degree */) 
    {
      init_loop_for<0,1>(r+1);
    }

    private:
      template<size_t l,size_t k> static void init_loop_for(int r) {
        if constexpr(k <= d) {
          if constexpr(l < k) {
            Diff_exterior<l,k>::init();
            init_loop_for<l+1,k>(r);
          } else {
            Monomial_powers<k>::init(r);
            init_loop_for<0,k+1>(r);
          }
        }
      }
  };
  ///@}

} // End namespace

#endif
