#version 330

in vec2 UVcoords;
in vec4 out_color;

out vec4 outColor;

uniform sampler2D grassTex;

void main() {
    outColor = texture(grassTex, UVcoords) * out_color;
}
