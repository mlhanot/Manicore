#include "maxwell.hpp"
#include "mesh_builder.hpp"
#include "sphere_ref.hpp"

#include <numbers>
#include <functional>

#include <fstream>
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

const char *meshfile = "../meshes/test/58_pts.json";
const char *mapfile = "meshes/test/libdisk_maps.so";
constexpr bool use_threads = true;

int main(int argc, char *argv[]) {
  // Parse options
  int degree;
  double dt;
  std::string logfile;
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("degree,d", po::value<int>(&degree)->default_value(0), "polynomial degree")
    ("step,t", po::value<double>(&dt)->default_value(1e-5), "time step")
    ("logfile,f", po::value<std::string>(), "file to write logs")
;
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }
  if (vm.count("logfile")) {
    logfile = vm["logfile"].as<std::string>();
    std::cout<<"Using \""<<logfile<<"\" as output"<<std::endl;
  } else {
    logfile = std::to_string(degree) + ".log";
    std::cout<<"No filename provided to output logs, defaulting to \""<<logfile<<"\""<<std::endl;
  }

  // Build the problem
  std::unique_ptr<Mesh<2>> mesh_ptr(Mesh_builder<2>::build(meshfile,mapfile));
  NullStream os;
  MaxwellProblem maxwell(*mesh_ptr,degree,dt,use_threads,nullptr,os);
  // Factorize the system
  maxwell.compute();

  // Select solution
  std::unique_ptr<Solution> sol(new Solution0());
  // Init 
  MaxwellVector uOld(maxwell);
  sol->_t = 0.;
  uOld.u.setZero();
  uOld.E() = maxwell.ddrcore().template interpolate<1>(std::bind_front(&Solution::E,sol.get()));
  uOld.B() = maxwell.ddrcore().template interpolate<2>(std::bind_front(&Solution::B,sol.get()));
  MaxwellVector u_h(maxwell);
  MaxwellVector uExact(maxwell);
  Eigen::VectorXd dEExact(maxwell.dimensionB());
  uExact.u.setZero();
  dEExact.setZero();
  // Time loop
  constexpr double tmax = 2.*std::numbers::pi;
  double t = 0., tprev = -1., tprint = 1e-2;
  std::vector<double> tHist, errE, errdE, errB;
  std::vector<double> normE, normdE, normB, normGh, normPhih;
  while(t < tmax) {
    // Backward Euler
    t += dt;
    sol->_t = t;
    Eigen::VectorXd rho_h = maxwell.ddrcore().template interpolate<0>(std::bind_front(&Solution::rho,sol.get()));
    Eigen::VectorXd J_h = maxwell.ddrcore().template interpolate<1>(std::bind_front(&Solution::J,sol.get()));
    maxwell.assembleRHS(-rho_h,dt*J_h - uOld.E(),uOld.B());
    u_h = maxwell.solve();
    // Process results
    uExact.E() = maxwell.ddrcore().template interpolate<1>(std::bind_front(&Solution::E,sol.get()));
    uExact.B() = maxwell.ddrcore().template interpolate<2>(std::bind_front(&Solution::B,sol.get()));
    dEExact = maxwell.ddrcore().template interpolate<2>(std::bind_front(&Solution::dE,sol.get()));
    tHist.push_back(t);
    errE.push_back(maxwell.norm(u_h.E()-uExact.E(),1));
    errdE.push_back(maxwell.normd(u_h.E(),dEExact,1));
    errB.push_back(maxwell.norm(u_h.B()-uExact.B(),2));
    normE.push_back(maxwell.norm(uExact.E(),1));
    normdE.push_back(maxwell.norm(dEExact,2));
    normB.push_back(maxwell.norm(uExact.B(),2));
    normGh.push_back(maxwell.norm(u_h.G(),0));
    normPhih.push_back(u_h.u[0]);
    if (t - tprev > tprint) {
      tprev = t;
      std::cout<<"Time: "<<t<<"\n";
      std::cout<<"Error E: "<<errE.back()<<", dE: "<<errdE.back()<<", B: "<<errB.back()<<"\n";
      std::cout<<"Norm E: "<<normE.back()<<", dE: "<<normdE.back()<<", B: "<<normB.back()<<"\n";
      std::cout<<"Norm G: "<<normGh.back()<<", phi: "<<normPhih.back()<<std::endl;
    }
    // Setup next iteration
    uOld = u_h;
  }
  // Write output
  std::fstream logfh{logfile, logfh.trunc | logfh.out};
  if (logfh.is_open()) {
    logfh << "# Using dt = "<<dt<<" and degree "<<degree<<std::endl;
    logfh << "# t\terrE\terrdE\terrB"<<std::endl;
    for (size_t i = 0; i < tHist.size(); ++i) {
      logfh<<tHist[i]<<"\t"<<errE[i]<<"\t"<<errdE[i]<<"\t"<<errB[i]<<"\n";
    }
    logfh.close();
  } else {
    std::cout<<"Cannot open logfile, skipping"<<std::endl;
  }

  return 0;
}

