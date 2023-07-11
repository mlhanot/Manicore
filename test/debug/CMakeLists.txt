ADD_EXECUTABLE(debug_graph graph.cpp)
TARGET_LINK_LIBRARIES(debug_graph mesh_builder)

ADD_EXECUTABLE(debug_dofs dofs.cpp)
TARGET_LINK_LIBRARIES(debug_dofs mesh_builder dofspace)

ADD_EXECUTABLE(debug_pullback pullback.cpp)
TARGET_LINK_LIBRARIES(debug_pullback mesh_builder)

ADD_EXECUTABLE(debug_integral integral.cpp)
TARGET_LINK_LIBRARIES(debug_integral mesh_builder quadrature)

ADD_EXECUTABLE(debug_orientation orientation.cpp)
TARGET_LINK_LIBRARIES(debug_orientation mesh_builder)
