#version 330

in vec2 UVcoords;

out vec4 outColor;

uniform sampler2D grassTex;

void main() {
    outColor = texture(grassTex, UVcoords) * vec4(0.180, 0.75, 0.341, 0);
}
