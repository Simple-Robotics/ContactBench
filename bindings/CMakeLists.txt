add_custom_target(python)
set_target_properties(python PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD True)

if(GENERATE_PYTHON_STUBS)
  include(${PROJECT_SOURCE_DIR}/cmake/stubs.cmake)
endif(GENERATE_PYTHON_STUBS)

# Name of the Python library
set(LIBRARY_NAME pycontact)


set(${LIBRARY_NAME}_HEADERS
  "contactbench.hh"
  "context.hh"
)

set(${LIBRARY_NAME}_SOURCES
  contactbench.cc
)

message(STATUS "Curr source dir: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Curr binary dir: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "HEADERS = ${${LIBRARY_NAME}_HEADERS}")

add_library(${LIBRARY_NAME} SHARED ${${LIBRARY_NAME}_SOURCES} ${${LIBRARY_NAME}_HEADERS})
target_include_directories(${LIBRARY_NAME} SYSTEM PRIVATE ${PYTHON_INCLUDE_DIRS})

add_dependencies(python ${LIBRARY_NAME})
add_header_group(${LIBRARY_NAME}_HEADERS)
add_source_group(${LIBRARY_NAME}_SOURCES)

target_link_boost_python(${LIBRARY_NAME} PUBLIC)
target_link_libraries(${LIBRARY_NAME} PUBLIC
  ${PROJECT_NAME}
  eigenpy::eigenpy
  Boost::system)

set_target_properties(${LIBRARY_NAME} PROPERTIES
  PREFIX "lib"
  SUFFIX "${PYTHON_EXT_SUFFIX}"
  LIBRARY_OUTPUT_NAME ${LIBRARY_NAME}
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bindings/${LIBRARY_NAME}"
  PUBLIC_HEADER "${${LIBRARY_NAME}_HEADERS}"
  VERSION ${PROJECT_VERSION}
  )

if(UNIX AND NOT APPLE)
  set_target_properties(${LIBRARY_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/../../..")
endif()

if(IS_ABSOLUTE ${PYTHON_SITELIB})
  set(ABSOLUTE_PYTHON_SITELIB ${PYTHON_SITELIB})
else()
  set(ABSOLUTE_PYTHON_SITELIB ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITELIB})
endif()
set(${LIBRARY_NAME}_INSTALL_DIR ${ABSOLUTE_PYTHON_SITELIB}/${LIBRARY_NAME})

set(PY_HEADERS_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/bindings)

install(TARGETS ${LIBRARY_NAME}
        PUBLIC_HEADER DESTINATION ${PY_HEADERS_INSTALL_DIR}
        LIBRARY DESTINATION ${${LIBRARY_NAME}_INSTALL_DIR})

# --- INSTALL SCRIPTS
set(PYTHON_FILES
  __init__.py
  solvers.py
  simulators.py
  inverse_dynamics.py
  utils/__init__.py
  utils/bench.py
  utils/pin_utils.py
  utils/callbacks.py
  torch/__init__.py
  torch/solvers.py
  jax/__init__.py
  jax/solvers.py
  )

if(GENERATE_PYTHON_STUBS)
  load_stubgen()
  generate_stubs(${CMAKE_BINARY_DIR}/python ${LIBRARY_NAME} ${ABSOLUTE_PYTHON_SITELIB})
endif(GENERATE_PYTHON_STUBS)

file(MAKE_DIRECTORY
     ${${LIBRARY_NAME}_INSTALL_DIR}/utils)

file(MAKE_DIRECTORY
     ${${LIBRARY_NAME}_INSTALL_DIR}/torch)

file(MAKE_DIRECTORY
     ${${LIBRARY_NAME}_INSTALL_DIR}/jax)

foreach(python ${PYTHON_FILES})
  PYTHON_BUILD(${LIBRARY_NAME} ${python})
  get_filename_component(pysubmodule ${python} PATH)
  get_filename_component(pyname ${python} NAME)
  set(MODULE_NAME ${LIBRARY_NAME}/${pysubmodule})

  python_install_on_site(${MODULE_NAME} ${pyname})
endforeach(python)

macro(create_symlink target linkname)
    execute_process(
        COMMAND ln -sf "${target}" "${linkname}"
        RESULT_VARIABLE HEADER_LINK_STATUS
        ERROR_VARIABLE HEADER_LINK_ERROR
    )
    if(NOT "${HEADER_LINK_STATUS}" EQUAL 0)
        message(FATAL_ERROR "Symlinking headers failed:\n${HEADER_LINK_ERROR}")
    endif()
endmacro()

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}/bindings)

if(BUILD_WITH_AUTODIFF_SUPPORT)
  target_compile_definitions(${LIBRARY_NAME} PUBLIC DIFFCONTACT_WITH_CPPAD)
  target_include_directories(${LIBRARY_NAME} PUBLIC ${cppad_INCLUDE_DIR})
endif(BUILD_WITH_AUTODIFF_SUPPORT)

create_symlink(
  ${CMAKE_CURRENT_SOURCE_DIR}/context.hh
  ${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}/bindings/context.hh)
