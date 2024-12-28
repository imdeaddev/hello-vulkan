cmake_minimum_required(VERSION 3.30)
project(vkstart)

set(CMAKE_CXX_STANDARD 23)

find_package(Vulkan QUIET REQUIRED)
add_subdirectory(deps/glfw)

add_executable(vkstart source/main.cpp)
target_link_libraries(vkstart PRIVATE Vulkan::Vulkan glfw)
