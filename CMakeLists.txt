cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(PRJ_NAME iris_detection)
project(${PRJ_NAME})
message(STATUS ${PROJECT_SOURCE_DIR})

set(IrisDet_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)
include_directories(${IrisDet_INCLUDE_DIRS})
message(STATUS ${IrisDet_INCLUDE_DIRS})

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin )
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin )

find_package(OpenCV REQUIRED)
set(LINK_LIBS "${OpenCV_LIBS}")
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB iris_detection_src
    "src/*.cpp"
)

add_executable (cli cli/main.cpp ${iris_detection_src})
target_link_libraries (cli ${OpenCV_LIBS})
