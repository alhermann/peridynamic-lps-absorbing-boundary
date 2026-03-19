cmake_minimum_required(VERSION 3.10)
project(PD2_LPS_ABC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include paths
include_directories(libraries/libkdtree++)
include_directories(src/helpers)

# Sources
set(SOURCES src/main.cpp)

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(${PROJECT_NAME} ${SOURCES})

if(OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

# Test data generator
add_executable(generate_test_data tests/generate_test_data.cpp)
set_target_properties(generate_test_data PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# ABC Preprocessor
add_executable(lps_abc_preprocessor preprocessor/main.cpp)
# No special compile definitions needed - using standard Fortran LAPACK interface
set_target_properties(lps_abc_preprocessor PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if(APPLE)
  target_link_libraries(lps_abc_preprocessor "-framework Accelerate")
else()
  find_package(LAPACK REQUIRED)
  target_link_libraries(lps_abc_preprocessor ${LAPACK_LIBRARIES})
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(lps_abc_preprocessor OpenMP::OpenMP_CXX)
endif()
