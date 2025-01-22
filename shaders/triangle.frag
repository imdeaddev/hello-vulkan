#version 450

layout(location = 0) in vec2 iTexCoord;

layout(location = 0) out vec4 oColor;

void main() {
    oColor = vec4(iTexCoord, 0.0, 1.0);
}
