cmake_minimum_required(VERSION 3.20)
project(winnex_madhava VERSION 1.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(MADHAVA_BUILD_TESTS "Build tests" ON)
option(MADHAVA_BUILD_PYTHON "Build Python bindings" ON)
option(MADHAVA_BUILD_BENCHMARK "Build benchmark executable" ON)
option(MADHAVA_BUILD_EXAMPLES "Build examples" ON)

# Dependencies
find_package(OpenMP REQUIRED)
# The Development component defines python3_add_library(), which pybind11's
# New-Tools macros require to create the _winnex_madhava module. It is safe in the
# manylinux container: headers ship under /opt/python/cp*-cp*/include and the
# pybind11 module build does not link against libpython itself.
find_package(Python3 COMPONENTS Interpreter Development)

# --- Main static library -----------------------------------------------
add_library(winnex_madhava
    src/winnex_madhava.cpp
)
target_include_directories(winnex_madhava PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(winnex_madhava PUBLIC OpenMP::OpenMP_CXX)

# --- Shared library (for pybind11) --------------------------------------
add_library(winnex_madhava_shared SHARED src/winnex_madhava.cpp)
set_target_properties(winnex_madhava_shared PROPERTIES OUTPUT_NAME winnex_madhava)
target_include_directories(winnex_madhava_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(winnex_madhava_shared PUBLIC OpenMP::OpenMP_CXX)

# Install
install(TARGETS winnex_madhava winnex_madhava_shared DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

# --- Python bindings -----------------------------------------------------
if(MADHAVA_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_winnex_madhava bindings/py_module.cpp)
    target_link_libraries(_winnex_madhava PRIVATE winnex_madhava)
    target_include_directories(_winnex_madhava PRIVATE include)
    # scikit-build-core: installs the extension into the python/winnex_madhava package dir
    install(TARGETS _winnex_madhava DESTINATION winnex_madhava)
endif()

# --- Tests ----------------------------------------------------------------
if(MADHAVA_BUILD_TESTS)
    enable_testing()
    add_executable(test_winnex_madhava tests/test_winnex_madhava.cpp)
    target_link_libraries(test_winnex_madhava PRIVATE winnex_madhava)
    target_include_directories(test_winnex_madhava PRIVATE include)
    add_test(NAME winnex_madhava COMMAND test_winnex_madhava)
endif()

# --- Benchmark -------------------------------------------------------------
if(MADHAVA_BUILD_BENCHMARK)
    add_executable(winnex_madhava_bench examples/bench_bigann_l2.cpp)
    target_link_libraries(winnex_madhava_bench PRIVATE winnex_madhava)
    target_include_directories(winnex_madhava_bench PRIVATE include)
endif()

# --- Examples ---------------------------------------------------------------
if(MADHAVA_BUILD_EXAMPLES)
    add_executable(example_minimal examples/example_minimal.cpp)
    target_link_libraries(example_minimal PRIVATE winnex_madhava)
    target_include_directories(example_minimal PRIVATE include)
endif()
