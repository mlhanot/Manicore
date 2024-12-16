This set of script is used to generate the data presented in the article. 

In order to run the test, copy the files of this folder into the build folder, and execute "run_testcase.sh".

This will generate the data for the smooth case. 
The other cases (C^0 solution on the sphere and torus) can be obtained following the same principles, but require an adjustment in the source code.

To compute the C^0 solution on the sphere, replace
  std::unique_ptr<Solution> sol(new SolutionL1());
by
  std::unique_ptr<Solution> sol(new Solution0());

in /Schemes/Maxwell2D/maxwell.cpp

To compute the C^0 solution on the torus, proceed as for the sphere, but define the preprosesor macro MAXWELLTORUS before compiling. 

