# GTest
# TODO: download this stuff
add_subdirectory(gtest-1.7.0)
include_directories(${gtest_SOURCE_DIR}/include)

enable_testing()

include_directories(${INFER_CPP_INCLUDE_DIRS})


add_executable(mathutil_test mathutil_test.cpp)
target_link_libraries(mathutil_test gtest_main infer-cpp)
add_test(mathutil_test mathutil_test)

add_executable(kmeans_test kmeans_test.cpp)
target_link_libraries(kmeans_test gtest_main infer-cpp)
add_test(kmeans_test kmeans_test)

add_executable(gmm_test gmm_test.cpp)
target_link_libraries(gmm_test gtest_main infer-cpp)
add_test(gmm_test gmm_test)

