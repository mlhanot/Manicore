ADD_EXECUTABLE(test_DDR_comm spaces_comm.cpp)
TARGET_LINK_LIBRARIES(test_DDR_comm mesh_builder ddr_spaces)

ADD_EXECUTABLE(test_DDR_exact spaces_exact.cpp)
TARGET_LINK_LIBRARIES(test_DDR_exact mesh_builder ddr_spaces)

ADD_EXECUTABLE(test_DDR_potential spaces_potential.cpp)
TARGET_LINK_LIBRARIES(test_DDR_potential mesh_builder ddr_spaces)

ADD_EXECUTABLE(test_DDR_interpolate spaces_interpolate.cpp)
TARGET_LINK_LIBRARIES(test_DDR_interpolate mesh_builder ddr_spaces)

ADD_CUSTOM_TARGET(test_DDR ALL
  DEPENDS test_DDR_comm test_DDR_exact test_DDR_potential test_DDR_interpolate)
ADD_CUSTOM_COMMAND(TARGET test_DDR POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                           ${CMAKE_CURRENT_SOURCE_DIR}/test_all.sh
                           ${CMAKE_CURRENT_BINARY_DIR}/test_all.sh)

