#version 330

in vec4 point;
in vec2 position;
in vec4 variance;
in vec2 scales;
in vec2 y_cos_sin;
in vec4 in_color;
out vec4 out_color;

uniform mat4 camera;

void main() {
    mat4 scaleMatrix = mat4(1.0);
    scaleMatrix[0][0] = scales.x;
    scaleMatrix[1][1] = scales.y;

    mat4 positionMatrix = mat4(1.0);
    positionMatrix[3][0] = position.x;
    positionMatrix[3][2] = position.y;

    mat4 rotation = mat4(1.0);
    rotation[0][0] = y_cos_sin.x;
    rotation[2][2] = y_cos_sin.x;
    rotation[2][0] = -y_cos_sin.y;
    rotation[0][2] = y_cos_sin.y;


    out_color = in_color;
	gl_Position = camera * (positionMatrix * rotation * scaleMatrix * point + positionMatrix * variance * point.y);
}
