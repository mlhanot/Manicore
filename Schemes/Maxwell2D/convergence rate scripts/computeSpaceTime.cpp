#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

int main (int argc, char*argv[]) {
  if (argc < 2) {
    std::cout<<"Usage: Provide the filename as first argument"<<std::endl;
    return 1;
  }
  size_t p = 2;
  if (argc > 2) {
    p = std::stoi(argv[2]);
  }
  std::vector<double> th,Eh,dEh,Bh;
  size_t lCount = 0;
  std::ifstream fs(argv[1], std::ios::in);
  for (std::string line; std::getline(fs, line);) {
    lCount++;
    if (line[0] == '#') continue;
    std::istringstream lineb(line);
    double t,E,dE,B;
    lineb >> t; 
    lineb >> E; 
    lineb >> dE; 
    lineb >> B; 
    if (lineb.fail()) {
      std::cerr<<"Invalid formatting on line "<<lCount<<". Skipping."<<std::endl;
      continue;
    }
    th.push_back(t);
    Eh.push_back(E);
    dEh.push_back(dE);
    Bh.push_back(B);
  }
  // Compute the time integral
  double EInt, dEInt, BInt;
  size_t nbItt = th.size();
  EInt = std::pow(Eh[0],p)*(th[1] - th[0])*0.5 + std::pow(Eh.at(nbItt-1),p)*(th.at(nbItt-1)-th.at(nbItt-2))*0.5;
  dEInt = std::pow(dEh[0],p)*(th[1] - th[0])*0.5 + std::pow(dEh.at(nbItt-1),p)*(th.at(nbItt-1)-th.at(nbItt-2))*0.5;
  BInt = std::pow(Bh[0],p)*(th[1] - th[0])*0.5 + std::pow(Bh.at(nbItt-1),p)*(th.at(nbItt-1)-th.at(nbItt-2))*0.5;
  for (size_t i = 1; i+1 < nbItt; ++i) {
    EInt += std::pow(Eh[i],p)*(th[i+1] - th[i-1])*0.5;
    dEInt += std::pow(dEh[i],p)*(th[i+1] - th[i-1])*0.5;
    BInt += std::pow(Bh[i],p)*(th[i+1] - th[i-1])*0.5;
  }
  EInt = std::pow(EInt,1./p);
  dEInt = std::pow(dEInt,1./p);
  BInt = std::pow(BInt,1./p);

  std::cout<<std::fixed;
  std::cout<<std::setprecision(10);
  std::cout<<"Error E: "<<EInt<<" Error dE: "<<dEInt<<" Error B: "<<BInt<<" Error: "<<EInt+dEInt+BInt<<std::endl;
  return 0;
}

