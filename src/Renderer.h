#pragma once

#include <glad/glad.h>
#include <vector>

class Renderer {
public:
    Renderer();
    ~Renderer();
    
    void render();
    
private:
    void initializeBuffers();
    void createShaders();
    void cleanup();
    
    GLuint m_VAO;
    GLuint m_VBO;
    GLuint m_shaderProgram;
};