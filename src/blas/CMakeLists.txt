cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(example_blas LANGUAGES CUDA)

add_executable(example_blas blas.cu)
target_link_libraries(example_blas PRIVATE cublas redscale)
