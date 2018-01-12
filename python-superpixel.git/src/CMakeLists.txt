cmake_minimum_required(VERSION 2.8)
project(gSLICrInterfaceProject)
  
IF(MSVC_IDE)
  set(OpenCV_STATIC OFF)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
ELSE(MSVC_IDE)
  set(CMAKE_CXX_FLAGS "-fPIC -O3 -shared ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
ENDIF(MSVC_IDE)

if(APPLE)
  set(CUDA_HOST_COMPILER /usr/bin/clang)
endif(APPLE)

set(GSLICR_HEADER_SRC
""
  CACHE PATH
  "Full Path to gSLICr Source Dir"
)

set(GSLICR_LIB_FULL_PATH
${GSLICR_HEADER_SRC}/build
  CACHE PATH
  "Full Path to libgSLICr_lib.a"
)

if(NOT EXISTS "${GSLICR_HEADER_SRC}/gSLICr_Lib/gSLICr.h")
  message(FATAL_ERROR "${GSLICR_HEADER_SRC}/gSLICr_Lib/gSLICr.h does not exists. Set the correct full path of gSLICr source with -DGSLICR_HEADER_SRC=.")
endif()

if(NOT EXISTS "${GSLICR_LIB_FULL_PATH}/libgSLICr_lib.a")
  message(FATAL_ERROR "${GSLICR_LIB_FULL_PATH}/LibgSLICr_Lib.a does not exists. Set the correct full path of generated libgSLICr_lib.a file with -DGSLICR_LIB_FULL_PATH=")
endif()

FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

FIND_PACKAGE(PythonLibs 2.7 REQUIRED)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

INCLUDE_DIRECTORIES(${GSLICR_HEADER_SRC})

SET(CMAKE_SWIG_FLAGS "")

SET_SOURCE_FILES_PROPERTIES(gSLICrInterface.i PROPERTIES CPLUSPLUS ON)

SWIG_ADD_MODULE(gSLICrInterface python gSLICrInterface.i gSLICrInterface.cpp conversion.cpp)
SWIG_LINK_LIBRARIES(gSLICrInterface ${GSLICR_LIB_FULL_PATH}/libgSLICr_lib.a ${PYTHON_LIBRARIES} ${OpenCV_LIBS} ${CUDA_LIBRARIES})

file(WRITE ${CMAKE_BINARY_DIR}/__init__.py "" )

