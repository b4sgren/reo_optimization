cmake_minimum_required(VERSION 3.4)
project(optimization_library)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED 1)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(include
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRW}
    )

IF(NOT TARGET gtest)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(libs/googletest)
ENDIF()

add_library(OPTIMIZATION STATIC
    reo.h
    reo.cpp
    structures.h
    )

add_executable(reo_test reo_test.cpp)
target_link_libraries(reo_test gtest gtest_main ${CERES_LIBRARIES} OPTIMIZATION)
