#pragma once

#include <vector>
#include <memory>
#include <cuda_runtime.h>

class CUDAGLInterop;

class CUDASimulationEngine {
public:
    CUDASimulationEngine();
    ~CUDASimulationEngine();
    
    // Initialize with OpenGL buffer for zero-copy
    bool initializeWithGL(unsigned int vbo, int rodPoints, float rodLength);
    
    // Initialize standalone (no GL interop)
    bool initialize(int rodPoints, float rodLength);
    
    // Cleanup
    void cleanup();
    
    // Update simulation
    void update(float deltaTime);
    
    // Reset to initial conditions
    void reset();
    
    // Parameter setters
    void setHeatSourceTemperature(float temp) { m_heatSourceTemp = temp; }
    void setHeatSourceTemp(float temp) { m_heatSourceTemp = temp; }  // Alias
    void setAmbientTemperature(float temp) { m_ambientTemp = temp; }
    void setAmbientTemp(float temp) { m_ambientTemp = temp; }  // Alias
    void setMaterialProperties(float conductivity, float density, float specificHeat);
    void setTimeStep(float dt) { m_timeStep = dt; }
    void setAutoTimeStep(bool auto_ts) { m_autoTimeStep = auto_ts; }
    
    // Color mapping
    void setColorScheme(int scheme) { m_colorScheme = scheme; }
    void setTemperatureRange(float min, float max) { m_minTemp = min; m_maxTemp = max; }
    
    // Getters
    const std::vector<float>& getTemperatures();
    float getSimulationTime() const { return m_simulationTime; }
    float getRodLength() const { return m_rodLength; }
    int getRodPoints() const { return m_rodPoints; }
    bool isStable() const;
    bool isUsingCUDA() const { return m_useCUDA; }
    
    // Control
    void pause() { m_isPaused = true; }
    void resume() { m_isPaused = false; }
    void setPaused(bool paused) { m_isPaused = paused; }
    bool isPaused() const { return m_isPaused; }
    
    // Performance metrics
    float getKernelTime() const { return m_kernelTime; }
    float getTransferTime() const { return m_transferTime; }
    size_t getMemoryUsage() const;
    
private:
    // CUDA resources
    float* d_temperature;        // Current temperature
    float* d_temperatureNew;     // Next temperature (double buffering)
    float* d_colorBuffer;        // Color data for rendering
    
    // Host data
    std::vector<float> h_temperature;
    
    // GL interop
    std::unique_ptr<CUDAGLInterop> m_glInterop;
    bool m_useGLInterop;
    unsigned int m_vbo;
    
    // Simulation parameters
    int m_rodPoints;
    float m_rodLength;
    float m_heatSourceTemp;
    float m_ambientTemp;
    
    // Material properties
    float m_thermalConductivity;
    float m_density;
    float m_specificHeat;
    
    // Simulation state
    float m_timeStep;
    float m_simulationTime;
    bool m_isPaused;
    bool m_autoTimeStep;
    bool m_useCUDA;
    
    // Visualization
    int m_colorScheme;
    float m_minTemp;
    float m_maxTemp;
    
    // Performance tracking
    float m_kernelTime;
    float m_transferTime;
    cudaEvent_t m_startEvent;
    cudaEvent_t m_stopEvent;
    
    // Helper methods
    bool allocateMemory();
    void freeMemory();
    float calculateStableTimeStep() const;
    void updateCPU(float deltaTime);  // Fallback
    void updateGPU(float deltaTime);  // CUDA implementation
};