cmake_minimum_required(VERSION 3.18)

project(numpy-iteration-examples)

if (MSVC)
    # warning level 4 and all warnings as errors
    add_compile_options(/W4 /WX)
else()
    # lots of warnings and all warnings as errors
    add_compile_options(-Wall -Wextra -pedantic)
endif()

if(WIN32)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE
        Release
        CACHE STRING "Build type" FORCE)
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "")
  link_directories(${Python3_LIBRARY_DIRS})
else()
  # we only choose Development.Module to support virtual environments where
  # libpython may not be available see
  # https://github.com/pypa/manylinux/issues/484
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
endif()

add_compile_definitions(NPY_NO_DEPRECATED_API=0)
python3_add_library(npyiters MODULE npyitersmodule.c)
target_include_directories(npyiters PUBLIC ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
