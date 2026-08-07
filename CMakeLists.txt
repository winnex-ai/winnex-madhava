cmake_minimum_required(VERSION 3.20)
project(winnex_madhava VERSION 1.0.1 LANGUAGES CXX)

# Optional CUDA for the native speed mode (cuBLAS QKᵀ matmul).
# Without a CUDA toolkit the CPU speed backend is used (OpenMP/AVX2).
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(MADHAVA_BUILD_TESTS "Build tests" ON)
option(MADHAVA_BUILD_PYTHON "Build Python bindings" ON)
option(MADHAVA_BUILD_BENCHMARK "Build benchmark executable" ON)
option(MADHAVA_BUILD_EXAMPLES "Build examples" ON)
# SIMD/optimization. AVX2+FMA are required for the fast paths (dot_f32,
# load_raw_norm). We enable them explicitly (not -march=native) so the
# manylinux wheel stays portable across x86-64 while still being fast.
# Machines without AVX2 fall back to the scalar path at runtime.
option(MADHAVA_ENABLE_SIMD "Enable AVX2/FMA + -O3" ON)

if(MADHAVA_ENABLE_SIMD AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -mfma -funroll-loops")
    add_compile_definitions(MADHAVA_FORCE_SIMD)
endif()

# Dependencies
find_package(OpenMP REQUIRED)
# The Development component defines python3_add_library(), which pybind11's
# New-Tools macros require to create the _winnex_madhava module. It is safe in the
# manylinux container: headers ship under /opt/python/cp*-cp*/include and the
# pybind11 module build does not link against libpython itself.
find_package(Python3 COMPONENTS Interpreter Development)

# --- Main static library -----------------------------------------------
set(MADHAVA_CORE_SRC src/winnex_madhava.cpp src/speed_cpu.cpp)
if(CMAKE_CUDA_COMPILER)
    list(APPEND MADHAVA_CORE_SRC src/speed_gpu.cu)
    add_compile_definitions(MADHAVA_HAS_CUDA)
endif()

add_library(winnex_madhava ${MADHAVA_CORE_SRC})
target_include_directories(winnex_madhava PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(winnex_madhava PUBLIC OpenMP::OpenMP_CXX)
if(CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(winnex_madhava PUBLIC CUDA::cublas CUDA::cudart)
endif()

# --- Shared library (for pybind11) --------------------------------------
add_library(winnex_madhava_shared SHARED ${MADHAVA_CORE_SRC})
set_target_properties(winnex_madhava_shared PROPERTIES OUTPUT_NAME winnex_madhava)
target_include_directories(winnex_madhava_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(winnex_madhava_shared PUBLIC OpenMP::OpenMP_CXX)
if(CMAKE_CUDA_COMPILER)
    target_link_libraries(winnex_madhava_shared PUBLIC CUDA::cublas CUDA::cudart)
endif()

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
