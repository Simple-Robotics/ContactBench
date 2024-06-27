macro(add_test_cflags target flag)
  set_property(TARGET "test-cpp-${target}" APPEND_STRING PROPERTY COMPILE_FLAGS " ${flag}")
endmacro(add_test_cflags)

macro(add_contact_bench_unit_test NAME)
  set(options HEADER_ONLY)

  set(PKGS ${unit_test_PACKAGES})

  set(TEST_NAME "test-cpp-${NAME}")
  add_unit_test(${TEST_NAME} ${NAME}.cpp)
  set_target_properties(${TEST_NAME} PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(${TEST_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  add_test_cflags(${NAME} "-DBOOST_TEST_DYN_LINK")
  set(MODULE_NAME "${NAME}Test")
  string(REGEX REPLACE "-" "_" MODULE_NAME ${MODULE_NAME})
  add_test_cflags(${NAME} "-DBOOST_TEST_MODULE=${MODULE_NAME}")
  if(WIN32)
    add_test_cflags(${NAME} "-DNOMINMAX -D_USE_MATH_DEFINES")
  endif(WIN32)

  target_link_libraries(${TEST_NAME} PUBLIC ${PROJECT_NAME})
  modernize_target_link_libraries(${TEST_NAME} SCOPE PRIVATE
    TARGETS Boost::unit_test_framework
    LIBRARIES ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
endmacro(add_contact_bench_unit_test)

find_package(Boost REQUIRED COMPONENTS unit_test_framework)

if(BUILD_WITH_AUTODIFF_SUPPORT)
  add_subdirectory(autodiff)
endif()

add_contact_bench_unit_test(friction-constraints)
add_contact_bench_unit_test(delassus)
add_contact_bench_unit_test(contact-problem)
add_contact_bench_unit_test(solvers)
add_contact_bench_unit_test(solvers-derivatives)


if(BUILD_PYTHON_INTERFACE)
  file(GLOB_RECURSE ${PROJECT_NAME}_PYTHON_UNITTEST *.py)

  foreach(TEST_FILE ${${PROJECT_NAME}_PYTHON_UNITTEST})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    string(REGEX REPLACE "${PROJECT_SOURCE_DIR}/" "" TEST_FILE ${TEST_FILE})
    add_python_unit_test("test-py-${TEST_NAME}" "${TEST_FILE}"
                         "bindings")
  endforeach(TEST_FILE ${${PROJECT_NAME}_PYTHON_UNITTEST})
endif(BUILD_PYTHON_INTERFACE)
