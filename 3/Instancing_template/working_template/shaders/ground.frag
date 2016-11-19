#version 330

in vec2 UVcoords;
out vec4 outColor;

uniform sampler2D groundTex;

void main() {
    outColor = vec4(0.333, 0.42, 0.184, 0);
    outColor = texture(groundTex, UVcoords);
}
