cmake_minimum_required(VERSION 3.16)
project( fht )

set(CMAKE_CXX_STANDARD 20)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( fht main.cpp fast_hough_transform.cpp)
target_link_libraries( fht ${OpenCV_LIBS} )
