cmake_minimum_required(VERSION 3.13)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(dawn)

find_package(Dawn REQUIRED)
add_executable(dawn main.cpp)

# Declare dependency on the dawn::webgpu_dawn library
target_link_libraries(dawn dawn::webgpu_dawn)

#your insall path here
#export CMAKE_PREFIX_PATH=/home/tom/Desktop/dawn/install/Release

#set config
#cmake -S . -B out/Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
