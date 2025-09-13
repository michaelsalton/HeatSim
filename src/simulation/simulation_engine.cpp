#include "simulation_engine.h"
#include "../utils/logger.h"
#include <algorithm>
#include <cmath>

SimulationEngine::SimulationEngine()
    : m_rodPoints(100)
    , m_rodLength(1.0f)
    , m_heatSourceTemp(100.0f)
    , m_ambientTemp(20.0f)
    , m_thermalConductivity(205.0f)  // Aluminum
    , m_density(2700.0f)
    , m_specificHeat(900.0f)
    , m_timeStep(0.001f)
    , m_simulationTime(0.0f)
    , m_isPaused(true) {
}

SimulationEngine::~SimulationEngine() {
}

void SimulationEngine::initialize(int rodPoints, float rodLength) {
    m_rodPoints = rodPoints;
    m_rodLength = rodLength;
    
    // Initialize temperature arrays
    m_temperatures.resize(m_rodPoints);
    m_tempBuffer.resize(m_rodPoints);
    
    reset();
    
    LOG_INFO("Simulation engine initialized with " + std::to_string(rodPoints) + " points");
}

void SimulationEngine::reset() {
    // Set initial temperature distribution
    for (int i = 0; i < m_rodPoints; ++i) {
        if (i == 0) {
            m_temperatures[i] = m_heatSourceTemp;
        } else {
            m_temperatures[i] = m_ambientTemp;
        }
    }
    
    m_tempBuffer = m_temperatures;
    m_simulationTime = 0.0f;
    
    LOG_DEBUG("Simulation reset to initial conditions");
}

void SimulationEngine::update(float deltaTime) {
    if (m_isPaused) return;
    
    // Use CPU implementation for Phase 1
    updateCPU(deltaTime);
    
    m_simulationTime += deltaTime;
}

void SimulationEngine::updateCPU(float deltaTime) {
    // Calculate thermal diffusivity
    float alpha = m_thermalConductivity / (m_density * m_specificHeat);
    float dx = m_rodLength / (m_rodPoints - 1);
    
    // Use provided timestep or calculate stable one
    float dt = (m_timeStep > 0) ? m_timeStep : calculateStableTimeStep();
    dt = std::min(dt, deltaTime);
    
    // Apply finite difference method
    for (int i = 1; i < m_rodPoints - 1; ++i) {
        float laplacian = (m_temperatures[i+1] - 2*m_temperatures[i] + m_temperatures[i-1]) / (dx * dx);
        m_tempBuffer[i] = m_temperatures[i] + alpha * dt * laplacian;
    }
    
    // Apply boundary conditions
    m_tempBuffer[0] = m_heatSourceTemp;  // Heat source
    m_tempBuffer[m_rodPoints - 1] = m_ambientTemp;  // Ambient end
    
    // Swap buffers
    std::swap(m_temperatures, m_tempBuffer);
}

float SimulationEngine::calculateStableTimeStep() const {
    float alpha = m_thermalConductivity / (m_density * m_specificHeat);
    float dx = m_rodLength / (m_rodPoints - 1);
    
    // CFL condition for stability: dt <= dx^2 / (2 * alpha)
    return 0.5f * (dx * dx) / alpha;
}

bool SimulationEngine::isStable() const {
    float requiredTimeStep = calculateStableTimeStep();
    return m_timeStep <= requiredTimeStep;
}

void SimulationEngine::setHeatSourceTemperature(float temp) {
    m_heatSourceTemp = temp;
    if (!m_temperatures.empty()) {
        m_temperatures[0] = temp;
    }
}

void SimulationEngine::setAmbientTemperature(float temp) {
    m_ambientTemp = temp;
    if (!m_temperatures.empty()) {
        m_temperatures[m_rodPoints - 1] = temp;
    }
}

void SimulationEngine::setMaterialProperties(float conductivity, float density, float specificHeat) {
    m_thermalConductivity = conductivity;
    m_density = density;
    m_specificHeat = specificHeat;
    
    // Recalculate timestep for stability
    if (m_timeStep > 0) {
        float stableTimeStep = calculateStableTimeStep();
        if (m_timeStep > stableTimeStep) {
            LOG_WARNING("Current timestep may be unstable. Consider reducing to " + 
                       std::to_string(stableTimeStep));
        }
    }
}

void SimulationEngine::setTimeStep(float dt) {
    m_timeStep = dt;
    
    if (!isStable()) {
        LOG_WARNING("Timestep " + std::to_string(dt) + 
                   " may be unstable. Maximum stable timestep: " + 
                   std::to_string(calculateStableTimeStep()));
    }
}