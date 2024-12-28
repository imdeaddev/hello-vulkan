cmake_minimum_required(VERSION 3.30)
project(vkstart)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

find_package(Vulkan QUIET REQUIRED)
add_subdirectory(deps/glfw)

add_executable(vkstart source/main.cpp)
target_link_libraries(vkstart PRIVATE Vulkan::Vulkan glfw)
