#include "cuda_gl_interop.h"
#include "cuda_manager.h"
#include "../utils/logger.h"

CUDAGLInterop::CUDAGLInterop()
    : m_initialized(false)
    , m_size(0)
    , m_vbo(0)
    , m_cudaResource(nullptr)
    , m_devicePtr(nullptr)
    , m_colorVBO(0)
    , m_colorResource(nullptr)
    , m_colorDevicePtr(nullptr)
    , m_colorSize(0)
    , m_isMapped(false)
    , m_isColorMapped(false) {
}

CUDAGLInterop::~CUDAGLInterop() {
    cleanup();
}

bool CUDAGLInterop::initialize(GLuint vbo, size_t size) {
    if (m_initialized) {
        cleanup();
    }
    
    m_vbo = vbo;
    m_size = size;
    
    // Register OpenGL buffer with CUDA
    cudaError_t error = cudaGraphicsGLRegisterBuffer(
        &m_cudaResource, 
        m_vbo, 
        cudaGraphicsMapFlagsWriteDiscard
    );
    
    if (error != cudaSuccess) {
        LOG_ERROR("Failed to register OpenGL buffer with CUDA: " + 
                  std::string(cudaGetErrorString(error)));
        return false;
    }
    
    m_initialized = true;
    LOG_INFO("CUDA-OpenGL interop initialized for buffer size: " + std::to_string(size));
    return true;
}

bool CUDAGLInterop::initializeColorBuffer(GLuint colorVBO, size_t numPoints) {
    m_colorVBO = colorVBO;
    m_colorSize = numPoints * 3 * sizeof(float);  // RGB per point
    
    cudaError_t error = cudaGraphicsGLRegisterBuffer(
        &m_colorResource,
        m_colorVBO,
        cudaGraphicsMapFlagsWriteDiscard
    );
    
    if (error != cudaSuccess) {
        LOG_ERROR("Failed to register color buffer with CUDA: " + 
                  std::string(cudaGetErrorString(error)));
        return false;
    }
    
    LOG_INFO("Color buffer registered for CUDA-GL interop");
    return true;
}

void CUDAGLInterop::cleanup() {
    if (m_isMapped) {
        unmapBuffer();
    }
    if (m_isColorMapped) {
        unmapColorBuffer();
    }
    
    if (m_cudaResource) {
        cudaGraphicsUnregisterResource(m_cudaResource);
        m_cudaResource = nullptr;
    }
    
    if (m_colorResource) {
        cudaGraphicsUnregisterResource(m_colorResource);
        m_colorResource = nullptr;
    }
    
    m_initialized = false;
    m_devicePtr = nullptr;
    m_colorDevicePtr = nullptr;
}

float* CUDAGLInterop::mapBuffer() {
    if (!m_initialized || m_isMapped) {
        return m_devicePtr;
    }
    
    // Map the resource
    cudaError_t error = cudaGraphicsMapResources(1, &m_cudaResource, 0);
    if (error != cudaSuccess) {
        LOG_ERROR("Failed to map CUDA resource: " + std::string(cudaGetErrorString(error)));
        return nullptr;
    }
    
    // Get device pointer
    size_t mappedSize;
    error = cudaGraphicsResourceGetMappedPointer(
        (void**)&m_devicePtr,
        &mappedSize,
        m_cudaResource
    );
    
    if (error != cudaSuccess) {
        LOG_ERROR("Failed to get mapped pointer: " + std::string(cudaGetErrorString(error)));
        cudaGraphicsUnmapResources(1, &m_cudaResource, 0);
        return nullptr;
    }
    
    m_isMapped = true;
    return m_devicePtr;
}

void CUDAGLInterop::unmapBuffer() {
    if (!m_isMapped) {
        return;
    }
    
    cudaError_t error = cudaGraphicsUnmapResources(1, &m_cudaResource, 0);
    if (error != cudaSuccess) {
        LOG_ERROR("Failed to unmap CUDA resource: " + std::string(cudaGetErrorString(error)));
    }
    
    m_isMapped = false;
    m_devicePtr = nullptr;
}

float* CUDAGLInterop::mapColorBuffer() {
    if (!m_colorResource || m_isColorMapped) {
        return m_colorDevicePtr;
    }
    
    cudaError_t error = cudaGraphicsMapResources(1, &m_colorResource, 0);
    if (error != cudaSuccess) {
        LOG_ERROR("Failed to map color resource: " + std::string(cudaGetErrorString(error)));
        return nullptr;
    }
    
    size_t mappedSize;
    error = cudaGraphicsResourceGetMappedPointer(
        (void**)&m_colorDevicePtr,
        &mappedSize,
        m_colorResource
    );
    
    if (error != cudaSuccess) {
        LOG_ERROR("Failed to get color mapped pointer: " + std::string(cudaGetErrorString(error)));
        cudaGraphicsUnmapResources(1, &m_colorResource, 0);
        return nullptr;
    }
    
    m_isColorMapped = true;
    return m_colorDevicePtr;
}

void CUDAGLInterop::unmapColorBuffer() {
    if (!m_isColorMapped) {
        return;
    }
    
    cudaError_t error = cudaGraphicsUnmapResources(1, &m_colorResource, 0);
    if (error != cudaSuccess) {
        LOG_ERROR("Failed to unmap color resource: " + std::string(cudaGetErrorString(error)));
    }
    
    m_isColorMapped = false;
    m_colorDevicePtr = nullptr;
}

void CUDAGLInterop::synchronize() {
    cudaDeviceSynchronize();
}