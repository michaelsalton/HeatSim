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
    
private:
    void initializeRodGeometry();
    void updateRodGeometry();
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
    
    // Matrices
    glm::mat4 m_projection;
    glm::mat4 m_view;
    
    int m_width;
    int m_height;
};