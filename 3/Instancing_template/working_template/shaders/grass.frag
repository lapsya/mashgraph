#version 330

in vec2 UVcoords;

out vec4 outColor;

uniform sampler2D grassTex;

void main() {
    outColor = texture(grassTex, UVcoords);
}
