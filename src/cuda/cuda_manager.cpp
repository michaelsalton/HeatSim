#include "cuda_manager.h"
#include "../utils/logger.h"
#include <sstream>
#include <iomanip>

CUDAManager::CUDAManager() 
    : m_initialized(false)
    , m_currentDevice(-1) {
}

CUDAManager::~CUDAManager() {
    cleanup();
}

bool CUDAManager::initialize() {
    if (m_initialized) {
        return true;
    }
    
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess || deviceCount == 0) {
        LOG_ERROR("No CUDA capable devices found");
        return false;
    }
    
    LOG_INFO("Found " + std::to_string(deviceCount) + " CUDA capable device(s)");
    
    // Select the best device (highest compute capability)
    int bestDevice = 0;
    int bestScore = 0;
    
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        
        int score = props.major * 1000 + props.minor * 10 + (props.totalGlobalMem / (1024 * 1024 * 1024));
        
        LOG_INFO("Device " + std::to_string(i) + ": " + std::string(props.name) + 
                " (Compute " + std::to_string(props.major) + "." + std::to_string(props.minor) + 
                ", " + std::to_string(props.totalGlobalMem / (1024 * 1024)) + " MB)");
        
        if (score > bestScore) {
            bestScore = score;
            bestDevice = i;
        }
    }
    
    if (!selectDevice(bestDevice)) {
        return false;
    }
    
    m_initialized = true;
    return true;
}

void CUDAManager::cleanup() {
    if (m_initialized) {
        cudaDeviceReset();
        m_initialized = false;
        LOG_INFO("CUDA manager cleaned up");
    }
}

bool CUDAManager::selectDevice(int deviceId) {
    cudaError_t error = cudaSetDevice(deviceId);
    if (error != cudaSuccess) {
        LOG_ERROR("Failed to set CUDA device " + std::to_string(deviceId));
        return false;
    }
    
    error = cudaGetDeviceProperties(&m_deviceProps, deviceId);
    if (error != cudaSuccess) {
        LOG_ERROR("Failed to get device properties");
        return false;
    }
    
    m_currentDevice = deviceId;
    
    LOG_INFO("Selected CUDA device: " + std::string(m_deviceProps.name));
    LOG_INFO("  Compute capability: " + std::to_string(m_deviceProps.major) + "." + 
             std::to_string(m_deviceProps.minor));
    LOG_INFO("  Total memory: " + std::to_string(m_deviceProps.totalGlobalMem / (1024 * 1024)) + " MB");
    LOG_INFO("  Multiprocessors: " + std::to_string(m_deviceProps.multiProcessorCount));
    LOG_INFO("  Max threads per block: " + std::to_string(m_deviceProps.maxThreadsPerBlock));
    
    return true;
}

int CUDAManager::getDeviceCount() const {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

std::string CUDAManager::getDeviceName() const {
    if (m_currentDevice < 0) return "No device selected";
    return std::string(m_deviceProps.name);
}

size_t CUDAManager::getDeviceMemory() const {
    if (m_currentDevice < 0) return 0;
    return m_deviceProps.totalGlobalMem;
}

int CUDAManager::getComputeCapability() const {
    if (m_currentDevice < 0) return 0;
    return m_deviceProps.major * 10 + m_deviceProps.minor;
}

bool CUDAManager::checkError(cudaError_t error, const std::string& function) {
    if (error != cudaSuccess) {
        std::stringstream ss;
        ss << "CUDA error in " << function << ": " 
           << cudaGetErrorName(error) << " - " 
           << cudaGetErrorString(error);
        LOG_ERROR(ss.str());
        return false;
    }
    return true;
}

void CUDAManager::synchronize() {
    cudaDeviceSynchronize();
}

size_t CUDAManager::getFreeMemory() const {
    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);
    return free;
}

size_t CUDAManager::getTotalMemory() const {
    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);
    return total;
}