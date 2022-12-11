cmake_minimum_required(VERSION 2.8)

project( FHT )

set(CMAKE_CXX_FLAGS "-O3 -Wall")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( FHT main.cpp fast_hough_transform.cpp)
target_link_libraries( FHT ${OpenCV_LIBS} )
