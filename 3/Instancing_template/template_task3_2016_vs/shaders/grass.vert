#version 330

in vec4 point;
//in vec2 position;
//in vec4 variance;

uniform mat4 camera;
uniform samplerBuffer tboSampler;
uniform samplerBuffer tboSampler2;


void main() {
    
    //vec2 bladePos = position;
    vec2 bladePos = texelFetch(tboSampler, gl_InstanceID).xy;
    vec4 var      = texelFetch(tboSampler2, gl_InstanceID);

    mat4 scaleMatrix = mat4(1.0);
    scaleMatrix[0][0] = 0.01;
    scaleMatrix[1][1] = 0.1;
    mat4 positionMatrix = mat4(1.0);
    positionMatrix[3][0] = bladePos.x;
    positionMatrix[3][2] = bladePos.y;

    gl_Position = camera * (positionMatrix * scaleMatrix * point + var * point.y);
}
