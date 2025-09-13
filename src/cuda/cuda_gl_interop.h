#pragma once

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glad/glad.h>
#include <vector>

class CUDAGLInterop {
public:
    CUDAGLInterop();
    ~CUDAGLInterop();
    
    // Initialize interop with OpenGL buffer
    bool initialize(GLuint vbo, size_t size);
    void cleanup();
    
    // Map/unmap for CUDA access
    float* mapBuffer();
    void unmapBuffer();
    
    // Direct color buffer interop
    bool initializeColorBuffer(GLuint colorVBO, size_t numPoints);
    float* mapColorBuffer();
    void unmapColorBuffer();
    
    // Synchronization
    void synchronize();
    
    // Getters
    bool isInitialized() const { return m_initialized; }
    size_t getSize() const { return m_size; }
    
private:
    bool m_initialized;
    size_t m_size;
    
    // Temperature buffer
    GLuint m_vbo;
    cudaGraphicsResource* m_cudaResource;
    float* m_devicePtr;
    
    // Color buffer
    GLuint m_colorVBO;
    cudaGraphicsResource* m_colorResource;
    float* m_colorDevicePtr;
    size_t m_colorSize;
    
    bool m_isMapped;
    bool m_isColorMapped;
};