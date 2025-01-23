#version 450

layout(location = 0) in vec3 iPosition;
layout(location = 1) in vec2 iTexCoord;

layout(binding = 0) uniform Scene {
    mat4 uView;
    mat4 uProj;
};

layout(push_constant) uniform PushConstants {
    mat4 uModel;
};

layout(location = 0) out vec2 oTexCoord;

void main() {
    gl_Position = uProj * uView * uModel * vec4(iPosition, 1.0);
    oTexCoord = iTexCoord;
}
