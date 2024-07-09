cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(example_ptx LANGUAGES CUDA)

add_executable(example_ptx ptx.cu)
