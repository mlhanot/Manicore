#include "maxwell.hpp"
#include "mesh_builder.hpp"
#include "sphere_ref.hpp"
#include "exporter.hpp"

#include <numbers>
#include <functional>

#include <fstream>
#include <limits>
#include <iomanip>
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
        _exporter(_maxwell->ddrcore().mesh(),_maxwell->ddrcore().degree(),15), 
        _outdir(outdir), _bad(false) {;}
    void save(int k, Eigen::Ref<const Eigen::VectorXd> const & u, const char* basename, int step) {
      if (_bad) return;
      std::string filename = _outdir + basename + "_" + std::to_string(step) + ".csv";
      if (_exporter.save(2-k,
            [&](size_t iT)->Eigen::VectorXd {
              return _maxwell->ddrcore().potential(k,2,iT)*_maxwell->ddrcore().dofspace(k).restrict(2,iT,u);},
            filename.c_str())) _bad = true;
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

const char *meshfile = "../meshes/test/58_pts.json";
const char *mapfile = "meshes/test/libdisk_maps.so";
constexpr bool use_threads = true;

int main(int argc, char *argv[]) {
  // Parse options
  int degree;
  double dt, t0, tmax, tprint;
  std::string logfile, outdir;
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("degree,d", po::value<int>(&degree)->default_value(0), "Polynomial degree")
    ("step,t", po::value<double>(&dt)->default_value(1e-5), "Time step")
    ("length,l", po::value<double>(&tmax)->default_value(2.*std::numbers::pi), "Simulation length")
    ("start", po::value<double>(&t0)->default_value(0.), "Starting time of the simulation")
    ("print,p", po::value<double>(&tprint)->default_value(1e-2), "Interval of simulation time between prints")
    ("logfile,f", po::value<std::string>(), "File to write logs")
    ("outdir,o", po::value<std::string>(), "Directory in which output the fields data")
;
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    std::cout << "Solve the 2 dimensional Maxwell on a sphere using a Crank-Nicolson time stepping \n"; 
    std::cout << desc << "\n";
    return 1;
  }
  if (vm.count("logfile")) {
    logfile = vm["logfile"].as<std::string>();
    std::cout<<"Using \""<<logfile<<"\" as output"<<std::endl;
  } else {
    logfile = "maxwell_d" + std::to_string(degree) + ".log";
    std::cout<<"No filename provided to output logs, defaulting to \""<<logfile<<"\""<<std::endl;
  }
  // Prepare logfile
  std::fstream logfh{logfile, logfh.trunc | logfh.out};
  if (logfh.is_open()) {
    logfh << std::setprecision(std::numeric_limits<double>::digits10+1);
    logfh << "# Using dt = "<<dt<<" and degree "<<degree<<std::endl;
    logfh << "# t\tE\tdE\tB\tG\tE^2+B^2"<<std::endl;
  } else {
    std::cerr<<"Cannot open logfile, skipping"<<std::endl;
  }

  // Build the problem
  std::unique_ptr<Mesh<2>> mesh_ptr(Mesh_builder<2>::build(meshfile,mapfile));
  NullStream os;
  MaxwellProblem maxwell(*mesh_ptr,degree,dt,use_threads,nullptr,os);
  std::shared_ptr<SaveCSV> saveCSV = nullptr; 
  if (vm.count("outdir")) {
    saveCSV = std::make_shared<SaveCSV>(&maxwell,vm["outdir"].as<std::string>().c_str());
  }
  // Factorize the system
  maxwell.compute();

  // Select solution
  std::unique_ptr<Solution> sol(new Solution1());
  // Initial values 
  double t = t0;
  sol->_t = t;
  Eigen::VectorXd J_hOld = maxwell.ddrcore().template interpolate<1>(std::bind_front(&Solution::J,sol.get()));
  MaxwellVector uOld(maxwell);
  uOld.u.setZero();
  uOld.E() = maxwell.ddrcore().template interpolate<1>(std::bind_front(&Solution::E,sol.get()));
  uOld.B() = maxwell.ddrcore().template interpolate<2>(std::bind_front(&Solution::B,sol.get()));
  // Print initial values if outdir is provided
  if (saveCSV) {
    saveCSV->save(2, uOld.B(), "B", 0);
    saveCSV->save(1, uOld.E(), "E", 0);
    saveCSV->saveNorm(1, uOld.E(), "NormE", 0);
  }
  {
    double normE = maxwell.norm(uOld.E(),1);
    double normdE = maxwell.normd(uOld.E(),1);
    double normB = maxwell.norm(uOld.B(),2);
    std::cout<<"T = "<<t<<" Initial values: E: "<<normE<<" dE: "<<normdE<<" B: "<<normB<<" E+B: "<<normE*normE+normB*normB<<std::endl;
    if (logfh.is_open()) {
      logfh <<t<<"\t"<<normE<<"\t"<<normdE<<"\t"<<normB<<"\t"<<0.<<"\t"<<normE*normE+normB*normB<<std::endl;
    }
  }
  // Declare containers
  MaxwellVector u_h(maxwell);
  int acc = 0;
  double tprev = -1.e151;

  // Time loop
  while(t < tmax) {
    t += dt;
    sol->_t = t;
    Eigen::VectorXd rho_h = maxwell.ddrcore().template interpolate<0>(std::bind_front(&Solution::rho,sol.get()));
    Eigen::VectorXd J_h = maxwell.ddrcore().template interpolate<1>(std::bind_front(&Solution::J,sol.get()));
    maxwell.assembleRHS(rho_h,0.5*(J_h + J_hOld),uOld.E(),uOld.B()); // CN
    u_h = maxwell.solve();
    // Setup next iteration
    uOld = u_h;
    J_hOld = J_h;
    // Process results
    double normG = maxwell.norm(u_h.G(),0);
    double normE = maxwell.norm(u_h.E(),1);
    double normdE = maxwell.normd(u_h.E(),1);
    double normB = maxwell.norm(u_h.B(),2);
    if (logfh.is_open()) {
      logfh <<t<<"\t"<<normE<<"\t"<<normdE<<"\t"<<normB<<"\t"<<0.<<"\t"<<normE*normE+normB*normB<<std::endl;
    }
    if (t - tprev > tprint) {
      tprev = t;
      acc++;
      std::cout<<"Time: "<<t<<"\n";
      std::cout<<"E+B: "<<normE*normE+normB*normB;
      std::cout<<" E: "<<normE<<" dE: "<<normdE<<" B: "<<normB<<" G: "<<normG<<std::endl;
      if (saveCSV) {
        saveCSV->save(2, uOld.B(), "B", acc);
        saveCSV->save(1, uOld.E(), "E", acc);
        saveCSV->saveNorm(1, uOld.E(), "NormE", acc);
      }
    }
  }

  return 0;
}

