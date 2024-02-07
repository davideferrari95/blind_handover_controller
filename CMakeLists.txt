cmake_minimum_required(VERSION 3.5)
project(handover_controller)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find Dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Find Messages Dependencies
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Install Python Modules
ament_python_install_package(scripts/utils)

# Install Python Scripts
install(PROGRAMS
  scripts/handover_controller.py
  scripts/experiments/goal_publisher.py
  scripts/experiments/ft_load_experiment.py
  scripts/experiments/save_data.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install Launch Files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install Config Files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  #set(ament_cmake_copyright_FOUND TRUE)
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
