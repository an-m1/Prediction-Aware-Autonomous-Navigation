cmake_minimum_required(VERSION 3.8)
project(vehicle_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Install directories and simulation files
install(
  DIRECTORY urdf launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES level1_world.sdf level1_map.yaml level1_map.pgm
  DESTINATION share/${PROJECT_NAME}
)

# Install executable python scripts
install(
  PROGRAMS run_level1.py astar.py rrt.py dwb.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
