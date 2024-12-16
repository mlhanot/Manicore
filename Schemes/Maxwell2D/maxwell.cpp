#include "maxwell.hpp"
#include "mesh_builder.hpp"

#ifdef MAXWELLTORUS
#include "torus_ref.hpp"
#else
#include "sphere_ref.hpp"
#endif
#include "exporter.hpp"

#include <numbers>
#include <functional>

#include <fstream>
#include <limits>
#include <iomanip>
#include <chrono>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

class NullStream : public std::ostream {
    class NullBuffer : public std::streambuf {
    public:
        int overflow( int c ) { return c; }
    } m_nb;
public:
    NullStream() : std::ostream( &m_nb ) {}
};

using namespace Manicore;

class SaveCSV {
  public:
    SaveCSV(MaxwellProblem const * maxwell, const char * outdir)
      : _maxwell(maxwell), 
        _exporter(_maxwell->ddrcore().mesh(),_maxwell->ddrcore().degree(),3), 
        _outdir(outdir), _bad(false) {;}
    void save(int k, Eigen::Ref<const Eigen::VectorXd> const & u, const char* basename, int step) {
      if (_bad) return;
      std::string filename = _outdir + basename + "_" + std::to_string(step) + ".csv";
      if (k == 1) {
        if (_exporter.save(2-k,
              [&](size_t iT)->Eigen::VectorXd {
              // ** = (-1)^{k(n-k)} = -1 for n = 2 and k = 1
                return -1.*_maxwell->ddrcore().potential(k,2,iT)*_maxwell->ddrcore().dofspace(k).restrict(2,iT,u);},
              filename.c_str(),true)) _bad = true;
      } else {
        if (_exporter.save(2-k,
              [&](size_t iT)->Eigen::VectorXd {
                return _maxwell->ddrcore().potential(k,2,iT)*_maxwell->ddrcore().dofspace(k).restrict(2,iT,u);},
              filename.c_str())) _bad = true;
      }
    }
    void saveNorm(int k, Eigen::Ref<const Eigen::VectorXd> const & u, const char* basename, int step) {
      if (_bad) return;
      std::string filename = _outdir + basename + "_" + std::to_string(step) + ".csv";
      if (_exporter.saveSq(2-k,
            [&](size_t iT)->Eigen::VectorXd {
              return _maxwell->ddrcore().potential(k,2,iT)*_maxwell->ddrcore().dofspace(k).restrict(2,iT,u);},
            filename.c_str())) _bad = true;
    }
  private:
    MaxwellProblem const * _maxwell;
    Exporter<2> _exporter;
    std::string _outdir;
    bool _bad;
};

#ifdef MAXWELLTORUS
//const char *mapfile = "meshes/torus/libtorus_3DEmbedding_shared.so";
const char *mapfile = "meshes/torus/libtorus_shared.so";
std::vector<const char *> meshfiles{"../meshes/torus/torus_5.json",
                                    "../meshes/torus/torus_10.json",
                                    "../meshes/torus/torus_15.json",
                                    "../meshes/torus/torus_20.json",
                                    "../meshes/torus/torus_25.json",
                                    "../meshes/torus/torus_30.json",
                                    "../meshes/torus/torus_35.json",
                                    "../meshes/torus/torus_40.json"};
#else
const char *mapfile = "meshes/sphere/libsphere_shared.so";
std::vector<const char *> meshfiles{"../meshes/sphere/4_circle.json",
                                    "../meshes/sphere/6_circle.json",
                                    "../meshes/sphere/11_circle.json",
                                    "../meshes/sphere/21_circle.json",
                                    "../meshes/sphere/29_circle.json",
                                    "../meshes/sphere/51_circle.json"};
#endif
                           
int main(int argc, char *argv[]) {
  // Parse options
  int degree;
  size_t meshNb;
  double dt, t0, tmax, tprint;
  bool use_threads, printSources, computeError, computeRhoPR;
  std::string logfile, outdir;
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("degree,d", po::value<int>(&degree)->default_value(0), "Polynomial degree")
    ("step,t", po::value<double>(&dt)->default_value(1e-5), "Time step")
    ("length,l", po::value<double>(&tmax)->default_value(2.*std::numbers::pi), "Simulation length")
    ("start", po::value<double>(&t0)->default_value(0.), "Starting time of the simulation")
    ("print,p", po::value<double>(&tprint)->default_value(1e-2), "Interval of simulation time between prints")
    ("print-extra", po::bool_switch(), "Print all available fields (include the norm of E, J, and rho)")
    ("logfile,f", po::value<std::string>(), "File to write logs")
    ("logfile-prefix", po::bool_switch(), "Use logfile as the prefix and append to automatic name")
    ("meshfile,m", po::value<size_t>(&meshNb)->default_value(2), "Index of the mesh in the sequence")
    ("outdir,o", po::value<std::string>(), "Directory in which output the fields data. If unset (default), no field data will be written")
    ("exact", po::bool_switch(), "Compute the error against the exact solution")
    ("disable-threads", po::bool_switch(), "Disable multithreading")
    ("WIP-compute-rho-pr", po::bool_switch(), "Check the preservation of the discrete rho. [WIP: Only for solutions with J = 0]")
;
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    std::cout << "Solve the 2 dimensional Maxwell on a sphere using a Crank-Nicolson time stepping \n"; 
    std::cout << desc << "\n";
    return 1;
  }
  // Set switches
  use_threads = not vm["disable-threads"].as<bool>(); // .count always return 1 when used with bool_switch
  printSources = vm["print-extra"].as<bool>();
  computeError = vm["exact"].as<bool>();
  computeRhoPR = vm["WIP-compute-rho-pr"].as<bool>();
  if (meshNb >= meshfiles.size()) {
    std::cout << "Meshfile number "<<meshNb<<" out of range. Please use a value less than "<<meshfiles.size()<<"\n";
    return 1;
  }
  logfile = "maxwell_m" + std::to_string(meshNb) + "_d" + std::to_string(degree) + ".log";
  if (vm.count("logfile")) {
    if (vm["logfile-prefix"].as<bool>()) {
      logfile = vm["logfile"].as<std::string>() + logfile;
    } else {
      logfile = vm["logfile"].as<std::string>();
    }
  }
  std::cout<<"Using \""<<logfile<<"\" as output"<<std::endl;
  // Prepare logfile
  std::fstream logfh{logfile, logfh.trunc | logfh.out};
  if (logfh.is_open()) {
    logfh << std::setprecision(std::numeric_limits<double>::digits10+1);
    logfh << "# Using dt = "<<dt<<" and degree "<<degree<<std::endl;
    logfh << "# Using the meshfile: "<<meshfiles.at(meshNb)<<std::endl;
  } else {
    std::cerr<<"Cannot open logfile, skipping"<<std::endl;
  }
  std::fstream logErrFh;
  if (computeError) {
    logErrFh.open((logfile.substr(0,logfile.size()-4) + "_err.log"), logErrFh.trunc | logErrFh.out);
    if (logErrFh.is_open()) {
    logErrFh << std::setprecision(std::numeric_limits<double>::digits10+1);
    logErrFh << "# Using dt = "<<dt<<" and degree "<<degree<<std::endl;
    logErrFh << "# Using the meshfile: "<<meshfiles.at(meshNb)<<std::endl;
    logErrFh << "# t\tE\tdE\tB"<<std::endl;
    } else {
      std::cerr<<"Cannot open logErrFile, skipping"<<std::endl;
    }
  }

  // Build the problem
  std::cout<<"Building the geometrical data"<<std::endl;
  std::unique_ptr<Mesh<2>> mesh_ptr(Mesh_builder<2>::build(meshfiles.at(meshNb),mapfile));
  NullStream os;
  MaxwellProblem maxwell(*mesh_ptr,degree,dt,use_threads,nullptr,os);
  std::shared_ptr<SaveCSV> saveCSV = nullptr; 
  if (vm.count("outdir")) {
    saveCSV = std::make_shared<SaveCSV>(&maxwell,vm["outdir"].as<std::string>().c_str());
  }
  // Mesh data
  std::cout<<"Mesh size: "<<maxwell.ddrcore().hmax()<<std::endl;
  // Factorize the system
  std::cout<<"Factorizing the system with "<<maxwell.dimensionSystem()<<" degrees of freedom"<<std::endl;
  maxwell.compute();

  // Select solution
  std::unique_ptr<Solution> sol(new SolutionL1());
  // Limite the number of write
  const unsigned errorRate = std::max(static_cast<int>((tmax-t0)/dt/5000),1);
  unsigned errorAcc = 0;
  std::cout<<"Expecting: "<<static_cast<int>((tmax-t0)/dt/errorRate)<<" error computation"<<std::endl;
  // Initial values 
  double t = t0;
  sol->_t = t;
  Eigen::VectorXd J_hOld = maxwell.ddrcore().template interpolate<1>(std::bind_front(&Solution::J,sol.get()));
  MaxwellVector uOld(maxwell);
  uOld.u.setZero();
  uOld.E() = maxwell.ddrcore().template interpolate<1>(std::bind_front(&Solution::E,sol.get()));
  uOld.B() = maxwell.ddrcore().template interpolate<2>(std::bind_front(&Solution::B,sol.get()));
  // Data to check the preservation of the discrete constraint
  Eigen::VectorXd E0;
  Eigen::SparseMatrix<double> Mdelta1;
  if (logfh.is_open() && computeRhoPR) {
    // delta = M_0^{-1}*d^T*M_1, we only compute d^T*M_1 to avoid solving another system each time 
    // The resulting value is not really the norm of delta, but it is zero iff delta is
    // For now, we do not consider a source, including one require to correctly integrate in time the corresponding source
    Mdelta1 = maxwell.globalDiff(0).transpose()*maxwell.globalMass(1);
    E0 = uOld.E();
  }
  // Print initial values if outdir is provided
  if (saveCSV) {
    saveCSV->save(2, uOld.B(), "B", 0);
    saveCSV->save(1, uOld.E(), "E", 0);
    if (printSources) {
      saveCSV->saveNorm(1, uOld.E(), "NormE", 0);
      saveCSV->save(1, J_hOld, "J", 0);
      saveCSV->save(0, maxwell.ddrcore().template interpolate<0>(std::bind_front(&Solution::rho,sol.get())), "rho", 0);
    }
  }
  {
    double normE = maxwell.norm(uOld.E(),1);
    double normdE = maxwell.normd(uOld.E(),1);
    double normB = maxwell.norm(uOld.B(),2);
    std::cout<<"T = "<<t<<" Initial values: E: "<<normE<<" dE: "<<normdE<<" B: "<<normB<<" E+B: "<<normE*normE+normB*normB<<std::endl;
    if (logfh.is_open()) {
      logfh << "# System dimension: "<<maxwell.dimensionSystem()<<std::endl;
      logfh << "# t\tE\tdE\tB\tG\tE^2+B^2"<<(computeRhoPR ? "\td^*E(t)-E(0)" : "")<<std::endl;
      logfh <<t<<"\t"<<normE<<"\t"<<normdE<<"\t"<<normB<<"\t"<<0.<<"\t"<<normE*normE+normB*normB;
      if (computeRhoPR) {
        logfh << "\t"<<0.; // Initial value, 0 by definition
      }
      logfh <<std::endl;
    }
    if (computeError){
      Eigen::VectorXd Eexact = maxwell.ddrcore().template interpolate<1>(std::bind_front(&Solution::E,sol.get()));
      Eigen::VectorXd Bexact = maxwell.ddrcore().template interpolate<2>(std::bind_front(&Solution::B,sol.get()));
      double errE = maxwell.norm(uOld.E() - Eexact,1);
      double errdE = maxwell.normd(uOld.E() - Eexact,1);
      double errB = maxwell.norm(uOld.B() - Bexact,2);
      std::cout<<"errE: "<<errE<<" errdE: "<<errdE<<" errB: "<<errB<<std::endl;
      if (logErrFh.is_open()) {
        logErrFh <<t<<"\t"<<errE<<"\t"<<errdE<<"\t"<<errB<<std::endl;
      }
    }
  }
  // Declare containers
  MaxwellVector u_h(maxwell);
  int acc = 0;
  double tprev = -1.e151;
  auto rtprev = std::chrono::steady_clock::now();

  // Time loop
  while(t < tmax) {
    t += dt;
    sol->_t = t;
#ifdef THREEFIELDS
    Eigen::VectorXd rho_h = maxwell.ddrcore().template interpolate<0>(std::bind_front(&Solution::rho,sol.get()));
#else
    Eigen::VectorXd rho_h;
#endif
    Eigen::VectorXd J_h;
    if (sol->_JZero) {
      J_h = Eigen::VectorXd::Zero(J_hOld.size());
    } else {
      J_h = maxwell.ddrcore().template interpolate<1>(std::bind_front(&Solution::J,sol.get()));
    }
    maxwell.assembleRHS(rho_h,0.5*(J_h + J_hOld),uOld.E(),uOld.B()); // CN
    u_h = maxwell.solve();
    // Setup next iteration
    uOld = u_h;
    J_hOld = J_h;
    // Process results
    if ((++errorAcc)%errorRate == 0) {
#ifdef THREEFIELDS
      double normG = maxwell.norm(u_h.G(),0);
#else
      double normG = 0;
#endif
      double normE = maxwell.norm(u_h.E(),1);
      double normdE = maxwell.normd(u_h.E(),1);
      double normB = maxwell.norm(u_h.B(),2);
      double errE, errdE, errB;
      if (computeError) {
          Eigen::VectorXd Eexact = maxwell.ddrcore().template interpolate<1>(std::bind_front(&Solution::E,sol.get()));
          Eigen::VectorXd Bexact = maxwell.ddrcore().template interpolate<2>(std::bind_front(&Solution::B,sol.get()));
          errE = maxwell.norm(uOld.E() - Eexact,1);
          errdE = maxwell.normd(uOld.E() - Eexact,1);
          errB = maxwell.norm(uOld.B() - Bexact,2);
      }
      if (logfh.is_open()) {
        logfh <<t<<"\t"<<normE<<"\t"<<normdE<<"\t"<<normB<<"\t"<<0.<<"\t"<<normE*normE+normB*normB;
        if (computeRhoPR) {
          double nRho = maxwell.norm(Mdelta1*(uOld.E() - E0),0);
          logfh <<"\t"<<nRho;
        }
        logfh <<std::endl;
        if (computeError && logErrFh.is_open()) {
          logErrFh <<t<<"\t"<<errE<<"\t"<<errdE<<"\t"<<errB<<std::endl;
        }
      }
      if (t - tprev > tprint) {
        tprev = t;
        auto rt = std::chrono::steady_clock::now();
        acc++;
        std::cout<<"Time: "<<t<<"\tClock timespan: "<<std::chrono::duration<double>(rt-rtprev).count()<<"\n";
        rtprev = rt;
        std::cout<<"E+B: "<<normE*normE+normB*normB;
        std::cout<<" E: "<<normE<<" dE: "<<normdE<<" B: "<<normB<<" G: "<<normG<<std::endl;
        if (computeError) {
          std::cout<<"errE: "<<errE<<" errdE: "<<errdE<<" errB: "<<errB<<std::endl;
        }
        if (saveCSV) {
          saveCSV->save(2, uOld.B(), "B", acc);
          saveCSV->save(1, uOld.E(), "E", acc);
          if (printSources) {
            saveCSV->saveNorm(1, uOld.E(), "NormE", acc);
            saveCSV->save(1, J_h, "J", acc);
#ifdef THREEFIELDS
            saveCSV->save(0, rho_h, "rho", acc);
#endif
          }
        }
      }
    }
  }

  return 0;
}

