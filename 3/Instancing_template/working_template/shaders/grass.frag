#version 330

in vec2 ex_position;
out vec4 outColor;

void main() {
    if (ex_position.x < 0.005 + 0.99 / 6) {
        outColor = vec4(1, 0, 0, 0);
    } else if (ex_position.x < 0.005 + 0.99 * 2 / 6){
        outColor = vec4(1, 0.5, 0, 0);
    } else if (ex_position.x < 0.005 + 0.99 * 3 / 6) {
        outColor = vec4(1, 1, 0, 0);
    } else if (ex_position.x < 0.005 + 0.99 * 4 / 6) {
        outColor = vec4(0, 1, 0, 0);
    } else if (ex_position.x < 0.005 + 0.99 * 5 / 6) {
        outColor = vec4(0, 0, 1, 0);
    } else {
        outColor = vec4(1, 0, 1, 0);
    }
    outColor = vec4(0.180, 0.545, 0.341, 0);
}
