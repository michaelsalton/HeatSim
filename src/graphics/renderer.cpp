#include "renderer.h"
#include "shader.h"
#include "../utils/logger.h"
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

Renderer::Renderer() 
    : m_rodVAO(0)
    , m_rodVBO(0)
    , m_rodEBO(0)
    , m_rodPoints(100)
    , m_colorScheme(0)
    , m_minTemp(0.0f)
    , m_maxTemp(100.0f)
    , m_width(1280)
    , m_height(720) {
    
    // Initialize with default temperatures
    m_temperatures.resize(m_rodPoints, 20.0f);
}

Renderer::~Renderer() {
    cleanup();
}

void Renderer::initialize() {
    LOG_INFO("Initializing renderer");
    
    // Load shaders from files
    std::string vertexSource, fragmentSource;
    
    // Read vertex shader
    std::ifstream vShaderFile("shaders/rod.vert");
    if (vShaderFile.is_open()) {
        std::stringstream vShaderStream;
        vShaderStream << vShaderFile.rdbuf();
        vertexSource = vShaderStream.str();
        vShaderFile.close();
    } else {
        LOG_WARNING("Could not open vertex shader file, using default");
        vertexSource = R"(
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
        )";
    }
    
    // Read fragment shader
    std::ifstream fShaderFile("shaders/rod.frag");
    if (fShaderFile.is_open()) {
        std::stringstream fShaderStream;
        fShaderStream << fShaderFile.rdbuf();
        fragmentSource = fShaderStream.str();
        fShaderFile.close();
    } else {
        LOG_WARNING("Could not open fragment shader file, using default");
        fragmentSource = R"(
            #version 430 core
            in float Temperature;
            out vec4 FragColor;
            uniform float minTemp;
            uniform float maxTemp;
            uniform int colorScheme;
            
            vec3 temperatureToColor(float t) {
                float normalized = clamp((t - minTemp) / (maxTemp - minTemp), 0.0, 1.0);
                vec3 color;
                if (normalized < 0.5) {
                    color = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), normalized * 2.0);
                } else {
                    color = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (normalized - 0.5) * 2.0);
                }
                return color;
            }
            
            void main() {
                vec3 color = temperatureToColor(Temperature);
                FragColor = vec4(color, 1.0);
            }
        )";
    }
    
    m_rodShader = std::make_unique<Shader>(vertexSource, fragmentSource);
    
    initializeRodGeometry();
    
    // Set initial view and projection
    m_view = glm::mat4(1.0f);
    resize(m_width, m_height);
    
    LOG_INFO("Renderer initialized successfully");
}

void Renderer::initializeRodGeometry() {
    // Create vertices for the rod (a horizontal bar)
    m_vertices.clear();
    m_indices.clear();
    
    float rodLength = 2.0f; // Normalized coordinates (-1 to 1)
    float rodHeight = 0.1f;
    
    // Create two rows of vertices (top and bottom of rod)
    for (int i = 0; i < m_rodPoints; ++i) {
        float x = -1.0f + (2.0f * i / (m_rodPoints - 1));
        float temp = m_temperatures[i];
        
        // Top vertex: x, y, temperature
        m_vertices.push_back(x);
        m_vertices.push_back(rodHeight);
        m_vertices.push_back(temp);
        
        // Bottom vertex: x, y, temperature
        m_vertices.push_back(x);
        m_vertices.push_back(-rodHeight);
        m_vertices.push_back(temp);
    }
    
    // Create indices for triangle strip
    for (int i = 0; i < m_rodPoints - 1; ++i) {
        int topLeft = i * 2;
        int bottomLeft = i * 2 + 1;
        int topRight = (i + 1) * 2;
        int bottomRight = (i + 1) * 2 + 1;
        
        // First triangle
        m_indices.push_back(topLeft);
        m_indices.push_back(bottomLeft);
        m_indices.push_back(topRight);
        
        // Second triangle
        m_indices.push_back(bottomLeft);
        m_indices.push_back(bottomRight);
        m_indices.push_back(topRight);
    }
    
    // Create OpenGL buffers
    glGenVertexArrays(1, &m_rodVAO);
    glGenBuffers(1, &m_rodVBO);
    glGenBuffers(1, &m_rodEBO);
    
    glBindVertexArray(m_rodVAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, m_rodVBO);
    glBufferData(GL_ARRAY_BUFFER, m_vertices.size() * sizeof(float), 
                 m_vertices.data(), GL_DYNAMIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_rodEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_indices.size() * sizeof(unsigned int),
                 m_indices.data(), GL_STATIC_DRAW);
    
    // Position attribute (location = 0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Temperature attribute (location = 1)
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 
                          (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
}

void Renderer::updateRodGeometry() {
    // Update vertex temperatures
    for (int i = 0; i < m_rodPoints; ++i) {
        float temp = (i < m_temperatures.size()) ? m_temperatures[i] : 20.0f;
        
        // Update top vertex temperature
        m_vertices[i * 6 + 2] = temp;
        // Update bottom vertex temperature
        m_vertices[i * 6 + 5] = temp;
    }
    
    // Update VBO
    glBindBuffer(GL_ARRAY_BUFFER, m_rodVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, m_vertices.size() * sizeof(float), 
                    m_vertices.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Renderer::render() {
    if (!m_rodShader) return;
    
    // Use shader and set uniforms
    m_rodShader->use();
    m_rodShader->setMat4("projection", m_projection);
    m_rodShader->setMat4("view", m_view);
    m_rodShader->setFloat("minTemp", m_minTemp);
    m_rodShader->setFloat("maxTemp", m_maxTemp);
    m_rodShader->setInt("colorScheme", m_colorScheme);
    
    // Draw the rod
    glBindVertexArray(m_rodVAO);
    glDrawElements(GL_TRIANGLES, m_indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void Renderer::resize(int width, int height) {
    m_width = width;
    m_height = height;
    
    // Update projection matrix
    float aspect = static_cast<float>(width) / static_cast<float>(height);
    m_projection = glm::ortho(-aspect, aspect, -1.0f, 1.0f, -1.0f, 1.0f);
}

void Renderer::setRodData(const std::vector<float>& temperatures) {
    m_temperatures = temperatures;
    m_rodPoints = temperatures.size();
    
    // Reinitialize geometry if size changed
    if (m_rodPoints * 2 != m_vertices.size() / 3) {
        initializeRodGeometry();
    } else {
        updateRodGeometry();
    }
}

void Renderer::cleanup() {
    if (m_rodVAO) {
        glDeleteVertexArrays(1, &m_rodVAO);
        m_rodVAO = 0;
    }
    if (m_rodVBO) {
        glDeleteBuffers(1, &m_rodVBO);
        m_rodVBO = 0;
    }
    if (m_rodEBO) {
        glDeleteBuffers(1, &m_rodEBO);
        m_rodEBO = 0;
    }
}