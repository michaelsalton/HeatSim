#include "ui_controller.h"
#include "../utils/logger.h"
#include "../cuda/cuda_manager.h"
#include <algorithm>
#include <cstring>
#include <imgui.h>

UIController::UIController() 
    : m_paramsChanged(false)
    , m_frameTime(0.0f)
    , m_fps(0.0f)
    , m_showAbout(false)
    , m_showPerformance(true)
    , m_showAdvanced(false)
    , m_selectedPreset(0) {
    
    m_lastParams = m_params;
    m_lastFrameTime = std::chrono::high_resolution_clock::now();
    
    // Initialize performance history
    m_fpsHistory.reserve(HISTORY_SIZE);
    m_frameTimeHistory.reserve(HISTORY_SIZE);
    
    // Initialize material presets
    m_materialPresets = {
        {"Aluminum", 205.0f, 2700.0f, 900.0f},
        {"Copper", 401.0f, 8960.0f, 385.0f},
        {"Iron", 80.0f, 7874.0f, 449.0f},
        {"Steel", 50.0f, 7850.0f, 490.0f},
        {"Gold", 317.0f, 19300.0f, 129.0f},
        {"Silver", 429.0f, 10500.0f, 235.0f},
        {"Brass", 109.0f, 8400.0f, 380.0f},
        {"Lead", 35.0f, 11340.0f, 129.0f},
        {"Titanium", 22.0f, 4500.0f, 523.0f},
        {"Glass", 1.0f, 2500.0f, 840.0f},
        {"Wood", 0.15f, 700.0f, 2300.0f},
        {"Concrete", 1.7f, 2300.0f, 880.0f}
    };
}

UIController::~UIController() {}

void UIController::render() {
    updatePerformanceMetrics();
    
    // Main control panel
    renderControlPanel();
    
    // Performance panel
    if (m_showPerformance) {
        renderPerformancePanel();
    }
    
    // About window
    if (m_showAbout) {
        renderAboutWindow();
    }
    
    // Check if parameters changed
    m_paramsChanged = std::memcmp(&m_params, &m_lastParams, sizeof(SimulationParams)) != 0;
    if (m_paramsChanged) {
        m_lastParams = m_params;
    }
}

void UIController::renderControlPanel() {
    ImGui::Begin("Heat Simulation Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    
    // Simulation controls
    ImGui::SeparatorText("Simulation Control");
    
    if (ImGui::Button(m_params.isPaused ? "Play" : "Pause", ImVec2(100, 0))) {
        togglePause();
        LOG_INFO(m_params.isPaused ? "Simulation paused" : "Simulation resumed");
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset", ImVec2(100, 0))) {
        reset();
        LOG_INFO("Simulation reset");
    }
    
    // Temperature controls
    ImGui::SeparatorText("Temperature Settings");
    
    ImGui::SliderFloat("Heat Source", &m_params.heatSourceTemp, 0.0f, 1000.0f, "%.1f°C");
    ImGui::SliderFloat("Ambient", &m_params.ambientTemp, -50.0f, 50.0f, "%.1f°C");
    
    if (ImGui::CollapsingHeader("Temperature Range")) {
        ImGui::DragFloatRange2("Display Range", &m_params.minTemp, &m_params.maxTemp, 
                               1.0f, -273.0f, 2000.0f, "Min: %.1f°C", "Max: %.1f°C");
    }
    
    // Rod properties
    ImGui::SeparatorText("Rod Properties");
    
    ImGui::SliderFloat("Length", &m_params.rodLength, 0.1f, 10.0f, "%.2f m");
    ImGui::SliderInt("Resolution", &m_params.rodPoints, 10, 1000, "%d points");
    
    // Material properties
    ImGui::SeparatorText("Material Properties");
    
    renderMaterialPresets();
    
    if (ImGui::CollapsingHeader("Custom Material")) {
        ImGui::DragFloat("Thermal Conductivity", &m_params.thermalConductivity, 
                         1.0f, 0.01f, 1000.0f, "%.2f W/m·K");
        ImGui::DragFloat("Density", &m_params.density, 
                         10.0f, 1.0f, 20000.0f, "%.0f kg/m³");
        ImGui::DragFloat("Specific Heat", &m_params.specificHeat, 
                         10.0f, 100.0f, 5000.0f, "%.0f J/kg·K");
    }
    
    // Simulation settings
    if (ImGui::CollapsingHeader("Advanced Settings", nullptr, 
                                m_showAdvanced ? ImGuiTreeNodeFlags_DefaultOpen : 0)) {
        m_showAdvanced = true;
        
        ImGui::Checkbox("Auto Time Step", &m_params.autoTimeStep);
        if (!m_params.autoTimeStep) {
            ImGui::DragFloat("Time Step", &m_params.timeStep, 
                            0.0001f, 0.00001f, 0.1f, "%.5f s", ImGuiSliderFlags_Logarithmic);
        }
        
        // Calculate thermal diffusivity
        float alpha = m_params.thermalConductivity / (m_params.density * m_params.specificHeat);
        ImGui::Text("Thermal Diffusivity: %.2e m²/s", alpha);
        
        // Stability check
        float dx = m_params.rodLength / m_params.rodPoints;
        float maxTimeStep = (dx * dx) / (2.0f * alpha);
        ImGui::Text("Max Stable Time Step: %.5f s", maxTimeStep);
        
        if (m_params.timeStep > maxTimeStep && !m_params.autoTimeStep) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Warning: Time step may be unstable!");
        }
    } else {
        m_showAdvanced = false;
    }
    
    // Visualization settings
    renderVisualizationSettings();
    
    // Window controls
    ImGui::Separator();
    ImGui::Checkbox("Show Performance", &m_showPerformance);
    ImGui::SameLine();
    if (ImGui::Button("About")) {
        m_showAbout = true;
    }
    
    ImGui::End();
}

void UIController::renderPerformancePanel() {
    ImGui::Begin("Performance Metrics", &m_showPerformance);
    
    ImGui::Text("FPS: %.1f", m_fps);
    ImGui::Text("Frame Time: %.3f ms", m_frameTime);
    
    // FPS Graph
    if (!m_fpsHistory.empty()) {
        ImGui::PlotLines("FPS", m_fpsHistory.data(), m_fpsHistory.size(), 
                        0, nullptr, 0.0f, 144.0f, ImVec2(0, 60));
    }
    
    // Frame time graph
    if (!m_frameTimeHistory.empty()) {
        ImGui::PlotLines("Frame Time (ms)", m_frameTimeHistory.data(), 
                        m_frameTimeHistory.size(), 0, nullptr, 0.0f, 33.3f, ImVec2(0, 60));
    }
    
    ImGui::Separator();
    ImGui::Text("Simulation Info:");
    ImGui::Text("Rod Points: %d", m_params.rodPoints);
    ImGui::Text("Time Step: %.5f s", m_params.timeStep);
    ImGui::Text("Status: %s", m_params.isPaused ? "Paused" : "Running");
    
    // CUDA Performance
    ImGui::Separator();
    ImGui::Text("CUDA Status:");
    if (CUDAManager::getInstance().isInitialized()) {
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "CUDA Active");
        ImGui::Text("Device: %s", CUDAManager::getInstance().getDeviceName().c_str());
        
        size_t freeMemory = CUDAManager::getInstance().getFreeMemory();
        size_t totalMemory = CUDAManager::getInstance().getTotalMemory();
        float memUsagePercent = 100.0f * (1.0f - (float)freeMemory / (float)totalMemory);
        
        ImGui::Text("Memory: %.1f%% used", memUsagePercent);
        ImGui::ProgressBar(memUsagePercent / 100.0f, ImVec2(-1, 0));
        ImGui::Text("Free: %.2f GB / Total: %.2f GB", 
                    freeMemory / (1024.0f * 1024.0f * 1024.0f),
                    totalMemory / (1024.0f * 1024.0f * 1024.0f));
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "CPU Mode");
        ImGui::Text("CUDA not available");
    }
    
    ImGui::End();
}

void UIController::renderMaterialPresets() {
    const char* presetNames[12];
    for (size_t i = 0; i < m_materialPresets.size(); ++i) {
        presetNames[i] = m_materialPresets[i].name.c_str();
    }
    
    if (ImGui::Combo("Material Preset", &m_selectedPreset, presetNames, m_materialPresets.size())) {
        const auto& preset = m_materialPresets[m_selectedPreset];
        m_params.thermalConductivity = preset.conductivity;
        m_params.density = preset.density;
        m_params.specificHeat = preset.specificHeat;
        LOG_INFO("Selected material preset: " + preset.name);
    }
    
    // Display current material properties
    ImGui::Text("k = %.1f W/m·K, ρ = %.0f kg/m³, c = %.0f J/kg·K",
                m_params.thermalConductivity, m_params.density, m_params.specificHeat);
}

void UIController::renderVisualizationSettings() {
    if (ImGui::CollapsingHeader("Visualization")) {
        const char* colorSchemes[] = { "Heat Map", "Grayscale", "Plasma" };
        ImGui::Combo("Color Scheme", &m_params.colorScheme, colorSchemes, 3);
        
        ImGui::Checkbox("Show Grid", &m_params.showGrid);
        ImGui::Checkbox("Show Legend", &m_params.showLegend);
    }
}

void UIController::renderAboutWindow() {
    ImGui::Begin("About Heat Sim", &m_showAbout, ImGuiWindowFlags_AlwaysAutoResize);
    
    ImGui::Text("Heat Sim v1.0.0");
    ImGui::Separator();
    ImGui::Text("A real-time 1D heat transfer simulation");
    ImGui::Text("Built with C++, CUDA, and OpenGL");
    ImGui::Separator();
    ImGui::Text("Controls:");
    ImGui::BulletText("ESC: Exit application");
    ImGui::BulletText("Space: Play/Pause simulation");
    ImGui::BulletText("R: Reset simulation");
    ImGui::Separator();
    ImGui::Text("© 2024 Heat Sim Project");
    
    if (ImGui::Button("Close")) {
        m_showAbout = false;
    }
    
    ImGui::End();
}

void UIController::updatePerformanceMetrics() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - m_lastFrameTime);
    m_lastFrameTime = now;
    
    m_frameTime = duration.count() / 1000.0f; // Convert to milliseconds
    m_fps = 1000000.0f / duration.count(); // Convert to FPS
    
    // Update history
    m_fpsHistory.push_back(m_fps);
    if (m_fpsHistory.size() > HISTORY_SIZE) {
        m_fpsHistory.erase(m_fpsHistory.begin());
    }
    
    m_frameTimeHistory.push_back(m_frameTime);
    if (m_frameTimeHistory.size() > HISTORY_SIZE) {
        m_frameTimeHistory.erase(m_frameTimeHistory.begin());
    }
}

void UIController::reset() {
    // Reset simulation-specific parameters
    m_params.isPaused = true;
    // Temperature distribution will be reset by the simulation engine
}