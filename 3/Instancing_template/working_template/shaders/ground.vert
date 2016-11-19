#version 330

in vec4 point;
out vec2 UVcoords;

uniform mat4 camera;

void main() {
    gl_Position = camera * point;
    UVcoords = vec2(point.x, point.z);
}
