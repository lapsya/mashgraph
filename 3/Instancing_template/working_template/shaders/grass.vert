#version 330

in vec4 point;
in vec2 position;
in vec4 variance;
in vec2 scales;
in vec2 y_cos_sin;
out vec2 UVcoords;


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


	gl_Position = camera * (positionMatrix * rotation * scaleMatrix * point + positionMatrix * variance * point.y);
    float scale = sqrt(point.x * point.x + point.y * point.y);
    if (point.x == 0 && point.y == 0) {
        UVcoords = vec2(0, 0);
    } else {
        UVcoords = vec2(point.x / sqrt(point.x * point.x + point.y * point.y), point.y / sqrt(point.x * point.x + point.y * point.y));
    }
}
