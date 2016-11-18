#version 330

in vec4 point;
in vec2 position;
in vec4 variance;
in vec2 scales;
out vec2 ex_position;

uniform mat4 camera;

void main() {
    mat4 scaleMatrix = mat4(1.0);
    scaleMatrix[0][0] = scales.x;
    scaleMatrix[1][1] = scales.y;
    mat4 positionMatrix = mat4(1.0);
    positionMatrix[3][0] = position.x;
    positionMatrix[3][2] = position.y;

	gl_Position = camera * (positionMatrix * scaleMatrix * point + variance * point.y);
    ex_position = position;
}
