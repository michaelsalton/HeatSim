#pragma once

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <string>
#include <memory>

class CUDAManager {
public:
    static CUDAManager& getInstance() {
        static CUDAManager instance;
        return instance;
    }
    
    bool initialize();
    void cleanup();
    
    // Device management
    int getDeviceCount() const;
    bool selectDevice(int deviceId);
    std::string getDeviceName() const;
    size_t getDeviceMemory() const;
    int getComputeCapability() const;
    
    // Error checking
    bool checkError(cudaError_t error, const std::string& function);
    void synchronize();
    
    // Memory info
    size_t getFreeMemory() const;
    size_t getTotalMemory() const;
    
    // Properties
    bool isInitialized() const { return m_initialized; }
    int getCurrentDevice() const { return m_currentDevice; }
    
private:
    CUDAManager();
    ~CUDAManager();
    
    bool m_initialized;
    int m_currentDevice;
    cudaDeviceProp m_deviceProps;
    
    // Prevent copying
    CUDAManager(const CUDAManager&) = delete;
    CUDAManager& operator=(const CUDAManager&) = delete;
};

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            CUDAManager::getInstance().checkError(error, #call); \
        } \
    } while(0)

#define CUDA_CHECK_RETURN(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            if (!CUDAManager::getInstance().checkError(error, #call)) { \
                return false; \
            } \
        } \
    } while(0)