ADD_EXECUTABLE(Maxwell2D maxwell.cpp)
TARGET_LINK_LIBRARIES(Maxwell2D mesh_builder ddr_spaces exporter Boost::program_options)

#ADD_EXECUTABLE(Maxwell2D_3fields maxwell.cpp)
#TARGET_LINK_LIBRARIES(Maxwell2D_3fields mesh_builder ddr_spaces exporter Boost::program_options)
#TARGET_COMPILE_DEFINITIONS(Maxwell2D_3fields PRIVATE THREEFIELDS)
