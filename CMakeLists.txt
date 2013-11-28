project(infer-cpp)
cmake_minimum_required(VERSION 2.8.8)

set(CMAKE_BUILD_TYPE Release)

# OSX
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "-std=c++11 -stdlib=libc++ -g -Wl,-no_pie")
    include_directories(${CMAKE_CURRENT_BINARY_DIR} /usr/local/include)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")


set(INFER_CPP_SOURCES
src/gmm.cpp
src/mathutil.cpp)

set(INFER_CPP_HEADERS
src/gmm.h
src/kmeans.h
src/mathutil.h)

set(INFER_CPP_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src)

# deps
find_package(armadillo REQUIRED)

add_library(infer-cpp ${INFER_CPP_SOURCES} ${INFER_CPP_HEADRS})
target_link_libraries(infer-cpp armadillo)

# tests
enable_testing()
add_subdirectory(test)
