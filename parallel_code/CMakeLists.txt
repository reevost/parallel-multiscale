#set(CMAKE_VERBOSE_MAKEFILE ON)

cmake_minimum_required(VERSION 3.20)
set(CUDAToolkit_ROOT "/opt/ohpc/pub/libs/cuda/11.1")
project(parallel_multiscale CUDA)


set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

enable_language(CXX)
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G")

add_executable(parallel_multiscale_times main_times.cu multiscale_structure.cuh iterative_parallel_methods.cuh wendland_functions.h wendland_functions_device.cuh macros.h)
#add_executable(parallel_multiscale main.cu multiscale_structure.cuh iterative_parallel_methods.cuh wendland_functions.h wendland_functions_device.cuh macros.h)
#add_executable(parallel_multiscale_build complete_approximant.cu)

find_package( BLAS REQUIRED )
message( STATUS BLAS found: ${BLAS_LIBRARIES})
message( STATUS BLAS found: ${BLAS_LINKER_FLAGS})

include_directories(/opt/ohpc/pub/libs/gnu9/openblas/0.3.7/include /opt/ohpc/pub/libs/cuda/11.1/targets/x86_64-linux/include)
link_directories(/opt/ohpc/pub/libs/gnu9/openblas/0.3.7/lib /opt/ohpc/pub/libs/cuda/11.1/targets/x86_64-linux/lib)

target_link_libraries(parallel_multiscale_times m)
#target_link_libraries(parallel_multiscale_build m)
#target_link_libraries(parallel_multiscale m)
target_link_libraries(parallel_multiscale_times ${BLAS_LIBRARIES} libcublas.so)
#target_link_libraries(parallel_multiscale ${BLAS_LIBRARIES} libcublas.so)

set_target_properties(parallel_multiscale_times PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(parallel_multiscale PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)

