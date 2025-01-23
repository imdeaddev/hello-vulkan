#version 450

layout(location = 0) in vec2 iTexCoord;

layout(push_constant) uniform PushConstants {
    layout(offset=64) vec4 uColor;
};

layout(location = 0) out vec4 oColor;

void main() {
    oColor = vec4(uColor.xyz, 1.0);
}
