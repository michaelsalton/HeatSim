#pragma once

#include <vector>
#include <memory>

// Forward declarations
class SimulationParams;

// This class will manage the heat simulation
// Currently a stub for Phase 1, will be implemented with CUDA in Phase 2
class SimulationEngine {
public:
    SimulationEngine();
    ~SimulationEngine();
    
    // Initialize simulation with given parameters
    void initialize(int rodPoints, float rodLength);
    
    // Update simulation by one timestep
    void update(float deltaTime);
    
    // Reset simulation to initial conditions
    void reset();
    
    // Set simulation parameters
    void setHeatSourceTemperature(float temp);
    void setAmbientTemperature(float temp);
    void setMaterialProperties(float conductivity, float density, float specificHeat);
    void setTimeStep(float dt);
    
    // Get current state
    const std::vector<float>& getTemperatures() const { return m_temperatures; }
    float getSimulationTime() const { return m_simulationTime; }
    bool isStable() const;
    
    // Control
    void pause() { m_isPaused = true; }
    void resume() { m_isPaused = false; }
    bool isPaused() const { return m_isPaused; }
    
private:
    // Simulation state
    std::vector<float> m_temperatures;
    std::vector<float> m_tempBuffer;  // Double buffering
    
    // Parameters
    int m_rodPoints;
    float m_rodLength;
    float m_heatSourceTemp;
    float m_ambientTemp;
    
    // Material properties
    float m_thermalConductivity;
    float m_density;
    float m_specificHeat;
    
    // Simulation control
    float m_timeStep;
    float m_simulationTime;
    bool m_isPaused;
    
    // Helper methods
    void updateCPU(float deltaTime);  // CPU implementation for Phase 1
    float calculateStableTimeStep() const;
};