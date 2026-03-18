cmake_minimum_required(VERSION 3.20)
project(quant-engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler warnings
add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion)

# Optimization for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address,undefined")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-fsanitize=address,undefined")

# Fetch dependencies
include(cmake/FetchDependencies.cmake)

# Core library
file(GLOB_RECURSE QE_SOURCES src/*.cpp)
add_library(qe ${QE_SOURCES})
target_include_directories(qe PUBLIC include)

# Tests
enable_testing()
add_subdirectory(tests)
