project(infer-cpp)
cmake_minimum_required(VERSION 2.8.8)

# for OSX only
set(CMAKE_CXX_FLAGS "-std=c++11 -stdlib=libc++ -g -Wl,-no_pie")

set(INFER_CPP_SOURCES
src/main.cpp
src/gmm.cpp)

set(INFER_CPP_HEADERS
src/gmm.h
src/kmeans.h)

set(INFER_CPP_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

#adsasd
# TODO: separate into a subproject?
# for building the gui
find_package(armadillo REQUIRED)

# homebrew includes for OSX
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    include_directories(${CMAKE_CURRENT_BINARY_DIR} /usr/local/include)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")


add_library(infer-cpp ${INFER_CPP_SOURCES})
target_link_libraries(infer-cpp armadillo)


enable_testing()
add_subdirectory(test)

