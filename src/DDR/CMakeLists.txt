ADD_LIBRARY(pec_integral pec.cpp)
TARGET_LINK_LIBRARIES(pec_integral integral_cells)

ADD_LIBRARY(ddr_spaces ddr_spaces.cpp)
TARGET_LINK_LIBRARIES(ddr_spaces pec_integral dofspace)

