#version 450

layout(location = 0) in vec2 iTexCoord;

layout(push_constant) uniform PushConstants {
    layout(offset=64) vec4 uColor;
};

layout(set = 1, binding = 0) uniform sampler2D uTexture;

layout(location = 0) out vec4 oColor;

void main() {
    oColor = texture(uTexture, iTexCoord);
}
