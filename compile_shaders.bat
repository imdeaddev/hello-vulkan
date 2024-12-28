@echo off
mkdir bin 2> nul
echo Compiling vertex shader
glslc shaders/triangle.vert -o bin/triangle.vert.spv
echo Compiling fragment shader
glslc shaders/triangle.frag -o bin/triangle.frag.spv