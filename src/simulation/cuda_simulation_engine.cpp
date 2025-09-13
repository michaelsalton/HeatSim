#include "cuda_simulation_engine.h"
#include "../cuda/cuda_manager.h"
#include "../cuda/cuda_gl_interop.h"
#include "../cuda/heat_kernels.cuh"
#include "../utils/logger.h"
#include <algorithm>
#include <cmath>
#include <cstring>

CUDASimulationEngine::CUDASimulationEngine()
    : d_temperature(nullptr)
    , d_temperatureNew(nullptr)
    , d_colorBuffer(nullptr)
    , m_useGLInterop(false)
    , m_vbo(0)
    , m_rodPoints(100)
    , m_rodLength(1.0f)
    , m_heatSourceTemp(100.0f)
    , m_ambientTemp(20.0f)
    , m_thermalConductivity(205.0f)
    , m_density(2700.0f)
    , m_specificHeat(900.0f)
    , m_timeStep(0.001f)
    , m_simulationTime(0.0f)
    , m_isPaused(true)
    , m_autoTimeStep(true)
    , m_useCUDA(false)
    , m_colorScheme(0)
    , m_minTemp(0.0f)
    , m_maxTemp(200.0f)
    , m_kernelTime(0.0f)
    , m_transferTime(0.0f) {
    
    // Create CUDA events for timing
    cudaEventCreate(&m_startEvent);
    cudaEventCreate(&m_stopEvent);
}

CUDASimulationEngine::~CUDASimulationEngine() {
    cleanup();
    cudaEventDestroy(m_startEvent);
    cudaEventDestroy(m_stopEvent);
}

bool CUDASimulationEngine::initializeWithGL(unsigned int vbo, int rodPoints, float rodLength) {
    m_vbo = vbo;
    m_useGLInterop = true;
    
    // Initialize CUDA if not already done
    if (!CUDAManager::getInstance().isInitialized()) {
        if (!CUDAManager::getInstance().initialize()) {
            LOG_ERROR("Failed to initialize CUDA manager");
            return false;
        }
    }
    
    m_rodPoints = rodPoints;
    m_rodLength = rodLength;
    
    // Setup GL interop
    m_glInterop = std::make_unique<CUDAGLInterop>();
    if (!m_glInterop->initialize(vbo, rodPoints * sizeof(float))) {
        LOG_ERROR("Failed to initialize CUDA-GL interop");
        return false;
    }
    
    // Allocate device memory
    if (!allocateMemory()) {
        return false;
    }
    
    reset();
    m_useCUDA = true;
    
    LOG_INFO("CUDA simulation engine initialized with GL interop");
    return true;
}

bool CUDASimulationEngine::initialize(int rodPoints, float rodLength) {
    m_useGLInterop = false;
    
    // Initialize CUDA if not already done
    if (!CUDAManager::getInstance().isInitialized()) {
        if (!CUDAManager::getInstance().initialize()) {
            LOG_WARNING("CUDA initialization failed, falling back to CPU");
            m_useCUDA = false;
            m_rodPoints = rodPoints;
            m_rodLength = rodLength;
            h_temperature.resize(rodPoints);
            reset();
            return true;
        }
    }
    
    m_rodPoints = rodPoints;
    m_rodLength = rodLength;
    
    // Allocate memory
    if (!allocateMemory()) {
        m_useCUDA = false;
        h_temperature.resize(rodPoints);
        reset();
        return true;
    }
    
    reset();
    m_useCUDA = true;
    
    LOG_INFO("CUDA simulation engine initialized (standalone mode)");
    return true;
}

bool CUDASimulationEngine::allocateMemory() {
    size_t tempSize = m_rodPoints * sizeof(float);
    size_t colorSize = m_rodPoints * 3 * sizeof(float);  // RGB
    
    // Allocate temperature buffers
    CUDA_CHECK_RETURN(cudaMalloc(&d_temperature, tempSize));
    CUDA_CHECK_RETURN(cudaMalloc(&d_temperatureNew, tempSize));
    
    // Allocate color buffer if not using GL interop
    if (!m_useGLInterop) {
        CUDA_CHECK_RETURN(cudaMalloc(&d_colorBuffer, colorSize));
    }
    
    // Allocate host buffer
    h_temperature.resize(m_rodPoints);
    
    size_t totalMemory = tempSize * 2 + (m_useGLInterop ? 0 : colorSize);
    LOG_INFO("Allocated " + std::to_string(totalMemory / 1024) + " KB of GPU memory");
    
    return true;
}

void CUDASimulationEngine::freeMemory() {
    if (d_temperature) {
        cudaFree(d_temperature);
        d_temperature = nullptr;
    }
    if (d_temperatureNew) {
        cudaFree(d_temperatureNew);
        d_temperatureNew = nullptr;
    }
    if (d_colorBuffer) {
        cudaFree(d_colorBuffer);
        d_colorBuffer = nullptr;
    }
}

void CUDASimulationEngine::cleanup() {
    if (m_glInterop) {
        m_glInterop->cleanup();
        m_glInterop.reset();
    }
    freeMemory();
    m_useCUDA = false;
}

void CUDASimulationEngine::reset() {
    // Initialize temperature array on host
    for (int i = 0; i < m_rodPoints; ++i) {
        if (i == 0) {
            h_temperature[i] = m_heatSourceTemp;
        } else {
            h_temperature[i] = m_ambientTemp;
        }
    }
    
    // Copy to device if using CUDA
    if (m_useCUDA && d_temperature) {
        CUDA_CHECK(cudaMemcpy(d_temperature, h_temperature.data(), 
                             m_rodPoints * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_temperatureNew, h_temperature.data(), 
                             m_rodPoints * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    m_simulationTime = 0.0f;
    LOG_DEBUG("Simulation reset");
}

void CUDASimulationEngine::update(float deltaTime) {
    if (m_isPaused) return;
    
    if (m_useCUDA) {
        updateGPU(deltaTime);
    } else {
        updateCPU(deltaTime);
    }
    
    m_simulationTime += deltaTime;
}

void CUDASimulationEngine::updateGPU(float deltaTime) {
    // Start timing
    cudaEventRecord(m_startEvent);
    
    // Calculate parameters
    float alpha = m_thermalConductivity / (m_density * m_specificHeat);
    float dx = m_rodLength / (m_rodPoints - 1);
    float dt = m_autoTimeStep ? calculateStableTimeStep() : m_timeStep;
    dt = std::min(dt, deltaTime);
    
    // Number of substeps
    int substeps = std::max(1, (int)(deltaTime / dt));
    dt = deltaTime / substeps;
    
    for (int step = 0; step < substeps; ++step) {
        // Launch heat diffusion kernel
        launchHeatDiffusionKernel(d_temperature, d_temperatureNew, alpha, dx, dt, m_rodPoints);
        
        // Apply boundary conditions
        launchBoundaryConditionsKernel(d_temperatureNew, m_heatSourceTemp, m_ambientTemp, m_rodPoints);
        
        // Swap buffers
        std::swap(d_temperature, d_temperatureNew);
    }
    
    // Update color buffer if using GL interop
    if (m_useGLInterop && m_glInterop) {
        float* d_color = m_glInterop->mapColorBuffer();
        if (d_color) {
            launchTemperatureToColorKernel(d_temperature, d_color, 
                                          m_minTemp, m_maxTemp, m_colorScheme, m_rodPoints);
            m_glInterop->unmapColorBuffer();
        }
    }
    
    // End timing
    cudaEventRecord(m_stopEvent);
    cudaEventSynchronize(m_stopEvent);
    cudaEventElapsedTime(&m_kernelTime, m_startEvent, m_stopEvent);
}

void CUDASimulationEngine::updateCPU(float deltaTime) {
    float alpha = m_thermalConductivity / (m_density * m_specificHeat);
    float dx = m_rodLength / (m_rodPoints - 1);
    float dt = m_autoTimeStep ? calculateStableTimeStep() : m_timeStep;
    dt = std::min(dt, deltaTime);
    
    std::vector<float> tempBuffer = h_temperature;
    
    // Apply finite difference method
    for (int i = 1; i < m_rodPoints - 1; ++i) {
        float laplacian = (h_temperature[i+1] - 2*h_temperature[i] + h_temperature[i-1]) / (dx * dx);
        tempBuffer[i] = h_temperature[i] + alpha * dt * laplacian;
    }
    
    // Boundary conditions
    tempBuffer[0] = m_heatSourceTemp;
    tempBuffer[m_rodPoints - 1] = m_ambientTemp;
    
    h_temperature = tempBuffer;
}

const std::vector<float>& CUDASimulationEngine::getTemperatures() {
    if (m_useCUDA && d_temperature) {
        // Copy from device to host
        cudaEventRecord(m_startEvent);
        CUDA_CHECK(cudaMemcpy(h_temperature.data(), d_temperature, 
                             m_rodPoints * sizeof(float), cudaMemcpyDeviceToHost));
        cudaEventRecord(m_stopEvent);
        cudaEventSynchronize(m_stopEvent);
        cudaEventElapsedTime(&m_transferTime, m_startEvent, m_stopEvent);
    }
    return h_temperature;
}

float CUDASimulationEngine::calculateStableTimeStep() const {
    // Calculate thermal diffusivity: α = k/(ρc)
    float alpha = m_thermalConductivity / (m_density * m_specificHeat);
    
    // Grid spacing
    float dx = m_rodLength / (m_rodPoints - 1);
    
    // CFL stability condition for explicit finite difference:
    // Δt_max = 0.5 * Δx² / α
    // We use 0.4 for safety margin (80% of theoretical maximum)
    return 0.4f * (dx * dx) / alpha;
}

bool CUDASimulationEngine::isStable() const {
    return m_timeStep <= calculateStableTimeStep();
}

void CUDASimulationEngine::setMaterialProperties(float conductivity, float density, float specificHeat) {
    m_thermalConductivity = conductivity;
    m_density = density;
    m_specificHeat = specificHeat;
    
    if (!m_autoTimeStep && !isStable()) {
        LOG_WARNING("Current timestep may be unstable with new material properties");
    }
}

size_t CUDASimulationEngine::getMemoryUsage() const {
    if (!m_useCUDA) return 0;
    
    size_t usage = m_rodPoints * sizeof(float) * 2;  // Two temperature buffers
    if (!m_useGLInterop) {
        usage += m_rodPoints * 3 * sizeof(float);  // Color buffer
    }
    return usage;
}