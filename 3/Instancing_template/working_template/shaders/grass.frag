#version 330

in vec4 out_color;
out vec4 outColor;

void main() {
    outColor = vec4(0.180, 0.545, 0.341, 0);
    outColor = out_color;
}
