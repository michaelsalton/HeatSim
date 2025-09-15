#pragma once

#include <imgui.h>
#include <string>
#include <vector>
#include <chrono>

struct SimulationParams {
    // Temperature settings
    float heatSourceTemp = 100.0f;
    float ambientTemp = 20.0f;
    float minTemp = 0.0f;
    float maxTemp = 200.0f;
    
    // Rod properties
    float rodLength = 1.0f;
    int rodPoints = 100;
    
    // Material properties
    float thermalConductivity = 205.0f;  // Aluminum W/m·K
    float density = 2700.0f;              // kg/m³
    float specificHeat = 900.0f;          // J/kg·K
    
    // Simulation settings
    float timeStep = 0.001f;
    bool isPaused = true;
    bool autoTimeStep = true;
    bool resetRequested = false;
    
    // Visual settings
    int colorScheme = 0;
    bool showGrid = true;
    bool showLegend = true;
};

class UIController {
public:
    UIController();
    ~UIController();
    
    void render();
    
    // Getters
    const SimulationParams& getParams() const { return m_params; }
    bool hasParamsChanged() const { return m_paramsChanged; }
    void resetParamsChanged() { m_paramsChanged = false; }
    
    // Control
    void togglePause() { m_params.isPaused = !m_params.isPaused; }
    void reset();
    void clearResetFlag() { m_params.resetRequested = false; }
    
private:
    void renderControlPanel();
    void renderPerformancePanel();
    void renderMaterialPresets();
    void renderVisualizationSettings();
    void renderAboutWindow();
    
    void updatePerformanceMetrics();
    
    SimulationParams m_params;
    SimulationParams m_lastParams;
    bool m_paramsChanged;
    
    // Performance tracking
    std::chrono::high_resolution_clock::time_point m_lastFrameTime;
    float m_frameTime;
    float m_fps;
    std::vector<float> m_fpsHistory;
    std::vector<float> m_frameTimeHistory;
    static constexpr int HISTORY_SIZE = 120;
    
    // UI state
    bool m_showAbout;
    bool m_showPerformance;
    bool m_showAdvanced;
    
    // Material presets
    struct MaterialPreset {
        std::string name;
        float conductivity;
        float density;
        float specificHeat;
    };
    std::vector<MaterialPreset> m_materialPresets;
    int m_selectedPreset;
};