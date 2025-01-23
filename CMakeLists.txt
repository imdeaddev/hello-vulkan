cmake_minimum_required(VERSION 3.30)
project(vkstart)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

find_package(Vulkan QUIET REQUIRED)
find_package(Stb REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Stb REQUIRED)

add_executable(vkstart source/main.cpp)
target_link_libraries(vkstart PRIVATE Vulkan::Vulkan glfw glm::glm)
target_include_directories(vkstart PRIVATE ${Stb_INCLUDE_DIR})
