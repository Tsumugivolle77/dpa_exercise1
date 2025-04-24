cmake_minimum_required(VERSION 3.10.0)
project(parallel_algo_demo VERSION 0.1.0 LANGUAGES C)

add_executable(parallel_algo_demo main.c)

add_compile_options(-fopenmp)

include_directories(/opt/homebrew/Caskroom/miniforge/base/include/)

find_package(OpenMP REQUIRED)
if(OpenMP_C_FOUND)
    target_link_libraries(parallel_algo_demo PUBLIC OpenMP::OpenMP_C)
endif()