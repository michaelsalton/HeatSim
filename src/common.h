#pragma once

// Common includes and definitions for Heat Sim project

// Standard library
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <chrono>

// OpenGL
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Math library
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// ImGui
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

// Project constants
namespace HeatSim {
    // Window defaults
    constexpr int DEFAULT_WIDTH = 1920;
    constexpr int DEFAULT_HEIGHT = 1080;
    
    // Simulation defaults
    constexpr int DEFAULT_ROD_POINTS = 100;
    constexpr float DEFAULT_ROD_LENGTH = 1.0f;
    constexpr float DEFAULT_AMBIENT_TEMP = 20.0f;
    constexpr float DEFAULT_HEAT_SOURCE_TEMP = 100.0f;
    
    // OpenGL version
    constexpr int GL_VERSION_MAJOR = 4;
    constexpr int GL_VERSION_MINOR = 3;
    
    // Physics constants
    constexpr float STEFAN_BOLTZMANN = 5.67e-8f;
    
    // Version info
    constexpr const char* VERSION = "1.0.0";
    constexpr const char* PROJECT_NAME = "Heat Sim";
}