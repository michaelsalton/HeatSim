#version 430 core
in float Temperature;
out vec4 FragColor;

uniform float minTemp;
uniform float maxTemp;
uniform int colorScheme;

vec3 temperatureToColor(float t) {
    // Normalize temperature to 0-1 range
    float normalized = clamp((t - minTemp) / (maxTemp - minTemp), 0.0, 1.0);
    
    vec3 color;
    if (colorScheme == 0) {
        // Heat map: blue -> cyan -> green -> yellow -> red
        if (normalized < 0.25) {
            float local = normalized * 4.0;
            color = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), local);
        } else if (normalized < 0.5) {
            float local = (normalized - 0.25) * 4.0;
            color = mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), local);
        } else if (normalized < 0.75) {
            float local = (normalized - 0.5) * 4.0;
            color = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), local);
        } else {
            float local = (normalized - 0.75) * 4.0;
            color = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), local);
        }
    } else if (colorScheme == 1) {
        // Grayscale
        color = vec3(normalized);
    } else {
        // Plasma colormap
        float r = sin(normalized * 3.14159);
        float g = sin(normalized * 3.14159 + 2.0);
        float b = cos(normalized * 3.14159);
        color = vec3(r, g * 0.7, b * 0.8);
    }
    
    return color;
}

void main() {
    vec3 color = temperatureToColor(Temperature);
    FragColor = vec4(color, 1.0);
}