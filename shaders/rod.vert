#version 430 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in float aTemperature;

out float Temperature;

uniform mat4 projection;
uniform mat4 view;

void main() {
    gl_Position = projection * view * vec4(aPos, 0.0, 1.0);
    Temperature = aTemperature;
}