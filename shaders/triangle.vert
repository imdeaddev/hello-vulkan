#version 450

const vec2 g_vertices[3] = vec2[](
    vec2(-0.5, -0.5),
    vec2(0.0, 0.5),
    vec2(0.5, -0.5)
);

void main() {
    gl_Position = vec4(g_vertices[gl_VertexIndex], 0.0, 1.0);
}
