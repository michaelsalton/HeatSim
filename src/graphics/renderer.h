#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <vector>
#include <memory>

class Shader;

class Renderer {
public:
    Renderer();
    ~Renderer();
    
    void initialize();
    void render();
    void resize(int width, int height);
    void setRodData(const std::vector<float>& temperatures);
    
    // Visual settings
    void setColorScheme(int scheme) { m_colorScheme = scheme; }
    void setTemperatureRange(float min, float max) { m_minTemp = min; m_maxTemp = max; }
    void setShowTemperatureValues(bool show) { m_showTemperatureValues = show; }
    void setTemperatureDisplayCount(int count) { m_temperatureDisplayCount = count; }
    
    // Camera controls
    void setZoom(float zoom);
    void pan(float deltaX, float deltaY);
    void resetCamera();
    
    // Get current temperatures for display
    const std::vector<float>& getTemperatures() const { return m_temperatures; }
    
private:
    void initializeRodGeometry();
    void updateRodGeometry();
    void renderTemperatureValues();
    void cleanup();
    
    // OpenGL objects
    GLuint m_rodVAO;
    GLuint m_rodVBO;
    GLuint m_rodEBO;
    
    // Shaders
    std::unique_ptr<Shader> m_rodShader;
    
    // Rod data
    std::vector<float> m_temperatures;
    std::vector<float> m_vertices;
    std::vector<unsigned int> m_indices;
    int m_rodPoints;
    
    // Visual settings
    int m_colorScheme;
    float m_minTemp;
    float m_maxTemp;
    bool m_showTemperatureValues = true;
    int m_temperatureDisplayCount = 5;
    
    // Matrices
    glm::mat4 m_projection;
    glm::mat4 m_view;
    
    // Camera
    float m_zoom = 1.0f;
    glm::vec2 m_pan = glm::vec2(0.0f, 0.0f);
    
    int m_width;
    int m_height;
};