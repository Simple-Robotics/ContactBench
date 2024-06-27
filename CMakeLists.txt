cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME contactbench)
set(PROJECT_DESCRIPTION "Contact benchmark")
set(PROJECT_URL "https://github.com/quentinll/diffcontact-bench")
set(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")
# set(CMAKE_CXX_STANDARD 11)
set(PROJECT_USE_KEYWORD_LINK_LIBRARIES TRUE)
set(INSTALL_DOCUMENTATION OFF CACHE BOOL ""  FORCE)

# Disable -Werror on Unix for now.
set(CXX_DISABLE_WERROR True)
set(CMAKE_VERBOSE_MAKEFILE True)

# Set CMake Policies
if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
endif(POLICY CMP0068)

# Check if the submodule cmake have been initialized
if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/cmake/base.cmake")
  message(
    FATAL_ERROR
      "\nPlease run the following command first:\ngit submodule update --init\n"
  )
endif()

# Include jrl-cmake submodules
include(${CMAKE_CURRENT_LIST_DIR}/cmake/base.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/ide.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/apple.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/boost.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/python.cmake)
include(CMakeDependentOption)
include(FetchContent)

compute_project_args(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})
set(CMAKE_VERBOSE_MAKEFILE OFF)

# If needed, set CMake policy for APPLE systems
apply_default_apple_configuration()

# Options
option(BUILD_PYTHON_INTERFACE "Build the python bindings" ON)
option(BUILD_WITH_AUTODIFF_SUPPORT "Build the library with the automatic differentiation support (via CppAD)" OFF)
option(GENERATE_PYTHON_STUBS "Generate the Python stubs associated to the Python library" OFF )
option(ENABLE_TEMPLATE_INSTANTIATION "Template instantiation of the main library" ON)
option(INITIALIZE_WITH_NAN "Initialize Eigen entries with NaN" OFF)
option(CHECK_RUNTIME_MALLOC "Check if some memory allocations are performed at runtime" OFF)

if(INITIALIZE_WITH_NAN)
  message(STATUS "Initialize with NaN all the Eigen entries.")
  add_definitions(-DEIGEN_INITIALIZE_MATRICES_BY_NAN)
endif(INITIALIZE_WITH_NAN)

if(ENABLE_TEMPLATE_INSTANTIATION)
  message(STATUS "Template instantiation of the main library")
  add_definitions(-DCONTACTBENCH_ENABLE_TEMPLATE_INSTANTIATION)
endif(ENABLE_TEMPLATE_INSTANTIATION)

if(CHECK_RUNTIME_MALLOC)
  Message(STATUS "Check if some memory allocations are performed at runtime.")
  Add_definitions(-DCONTACTBENCH_EIGEN_CHECK_MALLOC)
  Add_definitions(-DEIGEN_RUNTIME_NO_MALLOC)
endif(CHECK_RUNTIME_MALLOC)

set_boost_default_options()
export_boost_default_options()
if(WIN32)
  add_project_dependency(Boost REQUIRED COMPONENTS chrono thread date_time serialization)
elseif(WIN32)
  add_project_dependency(Boost REQUIRED chrono serialization)
endif(WIN32)

# MAIN LIBRARY
file(GLOB_RECURSE ${PROJECT_NAME}_HEADERS
     ${PROJECT_SOURCE_DIR}/include/contactbench/*.hpp
     ${PROJECT_SOURCE_DIR}/include/contactbench/*.hxx)

if(ENABLE_TEMPLATE_INSTANTIATION)
  set(LIB_TEMPLATE_SOURCES
    ${PROJECT_SOURCE_DIR}/src/solvers.cpp
    ${PROJECT_SOURCE_DIR}/src/contact-problem.cpp
    ${PROJECT_SOURCE_DIR}/src/delassus-wrapper.cpp
    )
  set(LIB_TEMPLATE_DECLARATIONS
    ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/solvers.txx
    ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/contact-problem.txx
    ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/delassus-wrapper.txx
    )
    list(APPEND ${PROJECT_NAME}_HEADERS ${LIB_TEMPLATE_DECLARATIONS})
    list(APPEND ${PROJECT_NAME}_SOURCES ${LIB_TEMPLATE_SOURCES})
endif()

add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS})
#adding dependencies
add_project_dependency(Eigen3 REQUIRED PKG_CONFIG_REQUIRES "eigen3 >= 3.0.5")
add_project_dependency(pinocchio REQUIRED)
add_project_dependency(proxsuite REQUIRED)
target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC
  PUBLIC Eigen3::Eigen
  PUBLIC pinocchio::pinocchio
  PUBLIC proxsuite::proxsuite
)
set_target_properties(${PROJECT_NAME}
  PROPERTIES
  LINKER_LANGUAGE CXX)

# Tests
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# Python bindings
if(BUILD_PYTHON_INTERFACE)
  add_project_dependency(eigenpy 2.9.0 REQUIRED)
  set(PYTHON_COMPONENTS Interpreter Development.Module)
  set(PY_NAME py${PROJECT_NAME})
  set(${PY_NAME}_INSTALL_DIR ${PYTHON_SITELIB}/${PY_NAME})

  add_project_dependency(Boost REQUIRED COMPONENTS system)
  add_subdirectory(bindings)
endif()

# target_compile_definitions(${PROJECT_NAME} PRIVATE -DPINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color")


# AD support
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/find-external/CppAD/" ${CMAKE_MODULE_PATH})
if(BUILD_WITH_AUTODIFF_SUPPORT)
  add_project_dependency(cppad 20180000.0 REQUIRED PKG_CONFIG_REQUIRES "cppad >= 20180000.0")
  target_compile_definitions(${PROJECT_NAME} PUBLIC DIFFCONTACT_WITH_CPPAD)
  target_link_libraries(${PROJECT_NAME} PUBLIC ${cppad_LIBRARY})
endif(BUILD_WITH_AUTODIFF_SUPPORT)

# Install library
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${TARGETS_EXPORT_NAME}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
