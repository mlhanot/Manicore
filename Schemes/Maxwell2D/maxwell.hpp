#ifndef MAXWELL_HPP_INCLUDED
#define MAXWELL_HPP_INCLUDED

#include <Eigen/Sparse>
#include <string_view>

#include "ddr_spaces.hpp"
#include "parallel_for.hpp"
#include <atomic>

#include <unsupported/Eigen/src/IterativeSolvers/Scaling.h>

/** @defgroup Schemes
  @brief Implement numerical schemes using this library
  */

/** \file maxwell.hpp
  Implement a scheme for the 2 dimensional Maxwell equation
  */

namespace Manicore {
  /// \addtogroup Schemes
  ///@{

  /// 2 dimensional Maxwell equation on a manifold without boundary
  /** Provides the logic to assemble and solve the system
      See the related paper.
    */
  class MaxwellProblem {
    public:
      /// Type used to store the system
      typedef Eigen::SparseMatrix<double> SystemMatrixType;
      /// Type used to store the solver
      typedef Eigen::SparseLU<SystemMatrixType,Eigen::COLAMDOrdering<int> > SolverType;
      /// Name of the solver
      static constexpr std::string_view SolverName = "SparseLU";

      MaxwellProblem(const Mesh<2> &mesh /*!< Mesh to use */,
                     int degree /*!< Polynomial degree */,
                     double timestep /*!< Time-step to use in the scheme */, 
                     bool use_threads = true /*!< Enable pthreads parallelism */, 
                     std::array<int,2> const* dqr = nullptr /*!< Degree of quadrature used to generate the mass matrices. It cannot be exact for generic metric and default to a pretty high degree. Use a lower degree if the metric and cells are almost flat. */,
                     std::ostream &output = std::cout /*!< Output stream for status messages. */);

      /// Return the number of unknowns attached to the space of 0 forms
      size_t dimensionG() const {return _ddrcore.dofspace(0).dimensionMesh();}
      /// Return the number of unknowns attached to the space of 1 forms
      size_t dimensionE() const {return _ddrcore.dofspace(1).dimensionMesh();}
      /// Return the number of unknowns attached to the space of 2 forms
      size_t dimensionB() const {return _ddrcore.dofspace(2).dimensionMesh();}
      /// Return the total number of unknowns
      size_t dimensionSystem() const {return 1 + dimensionG() + dimensionE() + dimensionB();}

      /// Return the associated DDR_Spaces
      DDR_Spaces<2> const & ddrcore() const {return _ddrcore;}
      /// Return the time-step
      double timeStep() const {return _dt;}

      /// Assemble the Right-Hand side from the given interpolates
      void assembleRHS(Eigen::Ref<const Eigen::VectorXd> const & rho /*!< Interpolate of the electric charge (0 form) */, 
                       Eigen::Ref<const Eigen::VectorXd> const & J /*!< Interpolate of the current (1 form)*/, 
                       Eigen::Ref<const Eigen::VectorXd> const & EOld /*!< Previous value of E_h */, 
                       Eigen::Ref<const Eigen::VectorXd> const & BOld /*!< Previous value of B_h*/);

      /// Setup the solver
      void compute();
      /// Check if the vector u if a solution up to a given relative accuracy
      bool validateSolution(Eigen::Ref<const Eigen::VectorXd> const &u) const;
      /// Solve the system and return the solution
      Eigen::VectorXd solve();
      /// Solve the system for the given Right-Hand side and return the solution
      Eigen::VectorXd solve(const Eigen::VectorXd &rhs);      

      /// Compute the discrete norm of a \f$k\f$-form
      /** 
        Compute \f$ \Vert E \Vert_{h,k} \f$ 
        */
      template<typename Derived>
      double norm(Eigen::MatrixBase<Derived> const &E /*!< Discrete form */, 
                  size_t k /*!< Form degree */) const;
      /// Compute the discrete norm of the differential of a \f$k\f$-form
      /** 
        Compute \f$ \Vert d_h^k E \Vert_{h,k+1} \f$ 
        */
      template<typename Derived>
      double normd(Eigen::MatrixBase<Derived> const &E /*!< Discrete form */, 
                   size_t k /*!< Form degree */) const;

    private:
      void assembleLocalContribution(Eigen::Ref<const Eigen::MatrixXd> const & A, size_t iT, size_t kL, size_t kR, std::forward_list<Eigen::Triplet<double>> * triplets) const;
      void assembleLocalContributionH(Eigen::Ref<const Eigen::MatrixXd> const & A, size_t iT, size_t k, std::forward_list<Eigen::Triplet<double>> * triplets) const;
      void assembleLocalContribution(Eigen::Ref<const Eigen::VectorXd> const & R, size_t iT, size_t k);
      // Assemble the system and initialize the masses matrices
      void assembleSystem(); 

    private:
      DDR_Spaces<2> _ddrcore;
      double _dt;
      bool _use_threads;
      std::ostream & _output;
      SystemMatrixType _system;
      Eigen::VectorXd _rhs;
      SolverType _solver;
      Eigen::VectorXd _interpOne; // Interpolate of the 0-form 1
      std::vector<Eigen::MatrixXd> _ALoc00; // Mass matrices
      std::vector<Eigen::MatrixXd> _ALoc11; // Mass matrices  
      std::vector<Eigen::MatrixXd> _ALoc22; // Mass matrices
      Eigen::VectorXd _scalingL, _scalingR; // Scaling to enhance the condition number
  };

  /// Construct a convenient wrapper around a vector of global unknowns to manipulate individual components
  struct MaxwellVector {
    MaxwellVector(MaxwellProblem const & maxwell /*!< Associated maxwell problem */)
      : u(maxwell.dimensionSystem()), sizeG(maxwell.dimensionG()), sizeE(maxwell.dimensionE()), sizeB(maxwell.dimensionB()),
      offsetG(1), offsetE(offsetG + sizeG), offsetB(offsetE + sizeE) {}
    /// Copy another MaxwellVector
    MaxwellVector& operator =(const MaxwellVector& other) {
      u = other.u;
      return *this;
    }
    /// Assign a vector of global unknowns
    MaxwellVector& operator =(const Eigen::VectorXd& other) {
      u = other;
      return *this;
    }
    /// Return the harmonic part
    Eigen::VectorBlock<Eigen::VectorXd> h() 
    {
      return u.head(1);
    }
    /// Return the ghost part
    Eigen::VectorBlock<Eigen::VectorXd> G()
    {
      return u.segment(offsetG,sizeG);
    }
    /// Return the electric part
    Eigen::VectorBlock<Eigen::VectorXd> E()
    {
      return u.segment(offsetE,sizeE);
    }
    /// Return the magnetic part
    Eigen::VectorBlock<Eigen::VectorXd> B()
    {
      return u.segment(offsetB,sizeB);
    }
    Eigen::VectorXd u;
    int sizeG,sizeE,sizeB;
    int offsetG,offsetE,offsetB;
  };
  ///@}

  MaxwellProblem::MaxwellProblem(const Mesh<2> &mesh,int degree, double dt, bool use_threads, std::array<int,2> const *dqr, std::ostream &output)
    : _ddrcore(mesh,degree,use_threads,dqr, output), _dt(dt), _use_threads(use_threads), _output(output), _rhs(dimensionSystem()) 
  {
    auto oneFunc = [](size_t , const Eigen::Vector<double,2> &)->Eigen::Vector<double,1> {
      return Eigen::Vector<double,1>{1.};
    };
    _interpOne = _ddrcore.template interpolate<0>(oneFunc,dqr);
    assembleSystem();
  }

  /// Helper to insert the local contribution of a cell
  void MaxwellProblem::assembleLocalContribution(Eigen::Ref<const Eigen::MatrixXd> const & A, size_t iT, size_t kL, size_t kR,
      std::forward_list<Eigen::Triplet<double>> * triplets) const
  {
    size_t gOffsetL = 1, gOffsetR = 1;
    // Initialise the global offsets
    switch(kL) {
      case 2:
        gOffsetL += dimensionE(); 
      case 1:
        gOffsetL += dimensionG();
      default:
        ;
    }
    switch(kR) {
      case 2:
        gOffsetR += dimensionE(); 
      case 1:
        gOffsetR += dimensionG();
      default:
        ;
    }
    for (size_t iDL = 0; iDL <= 2; ++iDL) { // Iterate dimensions on the left
      const size_t nbLDofsL = _ddrcore.dofspace(kL).numLocalDofs(iDL);
      if (nbLDofsL == 0) continue;
      std::vector<size_t> const & boundariesL = _ddrcore.mesh()->get_boundary(iDL,2,iT);
      for (size_t iDR = 0; iDR <= 2; ++iDR) { // Iterate dimensions on the right
        const size_t nbLDofsR = _ddrcore.dofspace(kR).numLocalDofs(iDR);
        if (nbLDofsR == 0) continue;
        std::vector<size_t> const & boundariesR = _ddrcore.mesh()->get_boundary(iDR,2,iT);
        for (size_t iFL = 0; iFL < boundariesL.size(); ++iFL) { // Iterate cells on the left
          const size_t offsetL = gOffsetL + _ddrcore.dofspace(kL).globalOffset(iDL,boundariesL[iFL]);
          const size_t lOffsetL = _ddrcore.dofspace(kL).localOffset(iDL,2,iFL,iT);
          for (size_t iFR = 0; iFR < boundariesR.size(); ++iFR) { // Iterate cells on the right
            const size_t offsetR = gOffsetR + _ddrcore.dofspace(kR).globalOffset(iDR,boundariesR[iFR]);
            const size_t lOffsetR = _ddrcore.dofspace(kR).localOffset(iDR,2,iFR,iT);
            for (size_t iRow = 0; iRow < nbLDofsL; ++iRow) { // Iterate local dofs 
              for (size_t iCol = 0; iCol < nbLDofsR; ++iCol) { // Iterate local dofs
                triplets->emplace_front(offsetL + iRow,offsetR + iCol,A(lOffsetL + iRow,lOffsetR + iCol));
              } // iCol
            } // iRow
          } // iFR
        } // iFL
      } // iDR
    } // iDL
  };
      
  void MaxwellProblem::assembleLocalContributionH(Eigen::Ref<const Eigen::MatrixXd> const & A, size_t iT, size_t k, std::forward_list<Eigen::Triplet<double>> * triplets) const
  {
    assert(A.cols() == 1 && "Expected only 1 harmonic form");
    size_t gOffset = 1;
    // Initialise the global offsets
    switch(k) {
      case 2:
        gOffset += dimensionE(); 
      case 1:
        gOffset += dimensionG();
      default:
        ;
    }
    for (size_t iD = 0; iD <= 2; ++iD) { // Iterate dimensions on the left
      const size_t nbLDofs = _ddrcore.dofspace(k).numLocalDofs(iD);
      if (nbLDofs == 0) continue;
      std::vector<size_t> const & boundaries = _ddrcore.mesh()->get_boundary(iD,2,iT);
      for (size_t iF = 0; iF < boundaries.size(); ++iF) { // Iterate cells on the left
        const size_t offset = gOffset + _ddrcore.dofspace(k).globalOffset(iD,boundaries[iF]);
        const size_t lOffset = _ddrcore.dofspace(k).localOffset(iD,2,iF,iT);
        for (size_t iRow = 0; iRow < nbLDofs; ++iRow) { // Iterate local dofs 
          triplets->emplace_front(offset+iRow,0,A(lOffset+iRow,0));
          triplets->emplace_front(0,offset+iRow,A(lOffset+iRow,0));
        } // iRow
      } // iF
    } // iD
  };
  
  void MaxwellProblem::assembleLocalContribution(Eigen::Ref<const Eigen::VectorXd> const & R, size_t iT, size_t k)
  {
    size_t gOffset = 1;
    // Initialise the global offsets
    switch(k) {
      case 2:
        gOffset += dimensionE(); 
      case 1:
        gOffset += dimensionG();
      default:
        ;
    }
    for (size_t iD = 0; iD <= 2; ++iD) { // Iterate dimensions on the left
      const size_t nbLDofs = _ddrcore.dofspace(k).numLocalDofs(iD);
      if (nbLDofs == 0) continue;
      std::vector<size_t> const & boundaries = _ddrcore.mesh()->get_boundary(iD,2,iT);
      for (size_t iF = 0; iF < boundaries.size(); ++iF) { // Iterate cells on the left
        const size_t offset = gOffset + _ddrcore.dofspace(k).globalOffset(iD,boundaries[iF]);
        const size_t lOffset = _ddrcore.dofspace(k).localOffset(iD,2,iF,iT);
        for (size_t iRow = 0; iRow < nbLDofs; ++iRow) { // Iterate local dofs 
          std::atomic_ref<double> atomicRHS(_rhs[offset+iRow]); // Prevent race condition when using several thread
          atomicRHS.fetch_add(R(lOffset+iRow),std::memory_order_relaxed); // Only requires the atomicity of the addition
        } // iRow
      } // iF
    } // iD
  };

  void MaxwellProblem::assembleSystem() 
  {
    const size_t nb_cell = _ddrcore.mesh()->n_cells(2);
    std::vector<Eigen::MatrixXd> ALoc0h(nb_cell);
    std::vector<Eigen::MatrixXd> ALoc01(nb_cell);
    std::vector<Eigen::MatrixXd> ALoc12(nb_cell);
    _ALoc00.resize(nb_cell);
    _ALoc11.resize(nb_cell);
    _ALoc22.resize(nb_cell);
    std::function<void(size_t start, size_t end)> assemble_local = [&](size_t start, size_t end)->void {
      for (size_t iT = start; iT < end; iT++) {
        Eigen::MatrixXd loc00 = _ddrcore.computeL2Product(0,2,iT);
        Eigen::MatrixXd loc11 = _ddrcore.computeL2Product(1,2,iT);
        Eigen::MatrixXd loc22 = _ddrcore.computeL2Product(2,2,iT);
        Eigen::MatrixXd loc0h = loc00*_ddrcore.dofspace(0).restrict(2,iT,_interpOne);
        Eigen::MatrixXd loc01 = _ddrcore.compose_diff(0,2,iT).transpose()*loc11;
        Eigen::MatrixXd loc12 = _ddrcore.compose_diff(1,2,iT).transpose()*loc22;
        ALoc0h[iT] = loc0h;
        ALoc01[iT] = loc01;
        ALoc12[iT] = loc12;
        _ALoc00[iT] = loc00;
        _ALoc11[iT] = loc11;
        _ALoc22[iT] = loc22;
      }
    };
    _output<< "[MaxwellProblem] Assembling local contributions..."<<std::flush;
    parallel_for(nb_cell,assemble_local,_use_threads);
    _output<<"\r[MaxwellProblem] Assembled local contributions    "<<std::endl;

    auto batch_local_assembly = [&](size_t start, size_t end, std::forward_list<Eigen::Triplet<double>> * triplets)->void {
      for (size_t iT = start; iT < end; iT++) {
        assembleLocalContributionH(ALoc0h[iT],iT,0,triplets);
        assembleLocalContribution(ALoc01[iT],iT,0,1,triplets); // Constraint <E,dv0>
        assembleLocalContribution(ALoc01[iT].transpose(),iT,1,0,triplets); // Constraint <dA,v1>
        assembleLocalContribution(0.5*_dt*ALoc12[iT],iT,1,2,triplets); // <dE,v2>
        assembleLocalContribution(0.5*_dt*ALoc12[iT].transpose(),iT,2,1,triplets); // <B,dv1>
        assembleLocalContribution(-1.*_ALoc11[iT],iT,1,1,triplets); // -<E,v1>
        assembleLocalContribution(_ALoc22[iT],iT,2,2,triplets); // <B,v2>
      }
    };
    _output<< "[MaxwellProblem] Assembling global system from local contributions..."<<std::flush;
    _system = parallel_assembly(nb_cell,std::make_pair(dimensionSystem(),dimensionSystem()),
        batch_local_assembly,_use_threads);
    _output<<"\r[MaxwellProblem] Assembled global system from local contributions    "<<std::endl;
  };
      
  void MaxwellProblem::assembleRHS(Eigen::Ref<const Eigen::VectorXd> const & rho, Eigen::Ref<const Eigen::VectorXd> const & J,Eigen::Ref<const Eigen::VectorXd> const & EOld, Eigen::Ref<const Eigen::VectorXd> const & BOld)
  {
    auto batch_local_assembly = [&](size_t start,size_t end)->void {
      for (size_t iT = start; iT < end; ++iT) {
        Eigen::VectorXd loc;
        // 0 forms
        assert(rho.size() == dimensionG() && "Incorrect size of rho");
        loc = -_ALoc00[iT]*_ddrcore.dofspace(0).restrict(2,iT,rho);
        assembleLocalContribution(loc,iT,0);
        // 1 forms
        assert(J.size() == dimensionE() && "Incorrect size of J");
        assert(EOld.size() == dimensionE() && "Incorrect size of EOld");
        loc = _ALoc11[iT]*_ddrcore.dofspace(1).restrict(2,iT,_dt*J - EOld);
        loc -= 0.5*_dt*_ddrcore.compose_diff(1,2,iT).transpose()*_ALoc22[iT]*_ddrcore.dofspace(2).restrict(2,iT,BOld);
        assembleLocalContribution(loc,iT,1);
        // 2 forms
        assert(BOld.size() == dimensionB() && "Incorrect size of BOld");
        loc = _ALoc22[iT]*_ddrcore.dofspace(2).restrict(2,iT,BOld);
        loc -= 0.5*_dt*_ALoc22[iT]*_ddrcore.compose_diff(1,2,iT)*_ddrcore.dofspace(1).restrict(2,iT,EOld);
        assembleLocalContribution(loc,iT,2);
      }
    };
    _output<< "[MaxwellProblem] Assembling RHS..."<<std::flush;
    _rhs.setZero();
    parallel_for(_ddrcore.mesh()->n_cells(2),batch_local_assembly,_use_threads);
    _output<<"\r[MaxwellProblem] Assembled RHS    "<<std::endl;
  };

  bool MaxwellProblem::validateSolution(Eigen::Ref<const Eigen::VectorXd> const &u) const 
  {
    double const err = (_system*u - _rhs).norm()/_rhs.norm();
    if (err > 1e-5) {
      std::cerr<<"[MaxwellProblem] The relative error on the solution is "<<err<<". This could indicate that the solver silently failed"<<std::endl;
      return false;
    }
    return true;
  }
  void MaxwellProblem::compute() {
    _output << "[MaxwellProblem] Setting solver "<<SolverName<<" with "<<dimensionSystem()<<" degrees of freedom"<<std::endl;
    SystemMatrixType systemScaled = _system;
    Eigen::IterScaling<SystemMatrixType> iterScaling;
    iterScaling.computeRef(systemScaled);
    _scalingL = iterScaling.LeftScaling();
    _scalingR = iterScaling.RightScaling();
    _solver.compute(systemScaled);
    if (_solver.info() != Eigen::Success) {
      std::cerr << "[MaxwellProblem] Failed to factorize the system" << std::endl;
      throw std::runtime_error("Factorization failed");
    }
  }
  Eigen::VectorXd MaxwellProblem::solve() {
    Eigen::VectorXd scaledRhs = _rhs.cwiseProduct(_scalingR);
    Eigen::VectorXd u = _solver.solve(scaledRhs);
    u = u.cwiseProduct(_scalingL);
    if (_solver.info() != Eigen::Success) {
      std::cerr << "[MaxwellProblem] Failed to solve the system" << std::endl;
      throw std::runtime_error("Solver failed");
    }
    validateSolution(u);
    return u;
  }

  Eigen::VectorXd MaxwellProblem::solve(const Eigen::VectorXd &rhs) {
    Eigen::VectorXd scaledRhs = rhs.cwiseProduct(_scalingR);
    Eigen::VectorXd u = _solver.solve(scaledRhs);
    u = u.cwiseProduct(_scalingL);
    if (_solver.info() != Eigen::Success) {
      std::cerr << "[MaxwellProblem] Failed to solve the system" << std::endl;
      throw std::runtime_error("Solver failed");
    }
    validateSolution(u);
    return u;
  }

  template<typename Derived>
  double MaxwellProblem::norm(Eigen::MatrixBase<Derived> const &E, size_t k) const
  {
    assert(E.size() == _ddrcore.dofspace(k).dimensionMesh() && "Mismatched dimension, wrong form degree?");
    const size_t nb_cell = _ddrcore.mesh()->n_cells(2);
    std::vector<double> accErr(nb_cell);
    std::function<void(size_t start, size_t end)> compute_local = [&](size_t start, size_t end)->void {
      for (size_t iT = start; iT < end; ++iT){
        Eigen::VectorXd locE = _ddrcore.dofspace(k).restrict(2,iT,E); 
        Eigen::MatrixXd matL2;
        switch(k) {
          case 0:
            matL2 = _ALoc00[iT];
            break;
          case 1:
            matL2 = _ALoc11[iT];
            break;
          case 2:
            matL2 = _ALoc22[iT];
            break;
          default:
            ;
        }
        accErr[iT] = locE.dot(matL2*locE);
      }
    };
    parallel_for(nb_cell,compute_local,_use_threads);
    double acc = 0;
    for (double err : accErr) {
      acc += err;
    }
    return std::sqrt(acc);
  }

  template<typename Derived>
  double MaxwellProblem::normd(Eigen::MatrixBase<Derived> const &E,size_t k) const
  {
    if (k == 2) return 0.;
    assert(E.size() == _ddrcore.dofspace(k).dimensionMesh() && "Mismatched dimension, wrong form degree?");
    const size_t nb_cell = _ddrcore.mesh()->n_cells(2);
    std::vector<double> accErr(nb_cell);
    std::function<void(size_t start, size_t end)> compute_local = [&](size_t start, size_t end)->void {
      for (size_t iT = start; iT < end; ++iT){
        Eigen::VectorXd locE = _ddrcore.compose_diff(k,2,iT)*_ddrcore.dofspace(k).restrict(2,iT,E);
        Eigen::MatrixXd matL2;
        switch(k) {
          case 0:
            matL2 = _ALoc11[iT];
            break;
          case 1:
            matL2 = _ALoc22[iT];
            break;
          default:
            ;
        }
        accErr[iT] = locE.dot(matL2*locE);
      }
    };
    parallel_for(nb_cell,compute_local,_use_threads);
    double acc = 0;
    for (double err : accErr) {
      acc += err;
    }
    return std::sqrt(acc);
  }

}; // Namespace
#endif

