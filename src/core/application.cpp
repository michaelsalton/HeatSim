#include "application.h"
#include "../graphics/renderer.h"
#include "../ui/ui_controller.h"
#include "../simulation/cuda_simulation_engine.h"
#include "../cuda/cuda_manager.h"
#include "../utils/logger.h"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <glad/glad.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

Application::Application(const std::string& title, int width, int height)
    : m_window(nullptr)
    , m_title(title)
    , m_width(width)
    , m_height(height)
    , m_running(false)
    , m_lastFrameTime(0.0f)
    , m_simulationTime(0.0f)
    , m_useCUDA(false) {
    init();
}

Application::~Application() {
    cleanup();
}

void Application::init() {
    // Initialize logger
    Logger::getInstance().setLogLevel(LogLevel::DEBUG);
    Logger::getInstance().enableFileLogging("heatsim.log");
    LOG_INFO("Initializing Heat Sim application");
    
    glfwSetErrorCallback(errorCallback);
    
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    
    m_window = glfwCreateWindow(m_width, m_height, m_title.c_str(), nullptr, nullptr);
    if (!m_window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    
    glfwMakeContextCurrent(m_window);
    glfwSetFramebufferSizeCallback(m_window, framebufferSizeCallback);
    glfwSetKeyCallback(m_window, keyCallback);
    glfwSetWindowUserPointer(m_window, this);
    
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        throw std::runtime_error("Failed to initialize GLAD");
    }
    
    glViewport(0, 0, m_width, m_height);
    glEnable(GL_DEPTH_TEST);
    
    glfwSwapInterval(1);
    
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    ImGui::StyleColorsDark();
    
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 430");
    
    m_renderer = std::make_unique<Renderer>();
    m_renderer->initialize();
    
    m_uiController = std::make_unique<UIController>();
    
    // Initialize CUDA simulation engine
    initializeCUDA();
    initializeSimulation();
    
    LOG_INFO("OpenGL Version: " + std::string((const char*)glGetString(GL_VERSION)));
    LOG_INFO("OpenGL Renderer: " + std::string((const char*)glGetString(GL_RENDERER)));
    LOG_INFO("Application initialized successfully");
}

void Application::cleanup() {
    LOG_INFO("Cleaning up application");
    
    m_uiController.reset();
    m_renderer.reset();
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    if (m_window) {
        glfwDestroyWindow(m_window);
    }
    glfwTerminate();
}

void Application::run() {
    m_running = true;
    m_lastFrameTime = static_cast<float>(glfwGetTime());
    
    while (m_running && !glfwWindowShouldClose(m_window)) {
        float currentTime = static_cast<float>(glfwGetTime());
        float deltaTime = currentTime - m_lastFrameTime;
        m_lastFrameTime = currentTime;
        
        glfwPollEvents();
        processInput();
        
        update(deltaTime);
        render();
        
        glfwSwapBuffers(m_window);
    }
}

void Application::processInput() {
    if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(m_window, true);
    }
}

void Application::update(float deltaTime) {
    // Update simulation
    updateSimulation(deltaTime);
    
    // Check for UI parameter changes
    if (m_uiController && m_uiController->hasParamsChanged()) {
        const auto& params = m_uiController->getParams();
        
        // Update renderer settings
        m_renderer->setColorScheme(params.colorScheme);
        m_renderer->setTemperatureRange(params.minTemp, params.maxTemp);
        
        // Resize temperature array if needed
        if (params.rodPoints != m_temperatures.size()) {
            initializeSimulation();
        }
        
        m_uiController->resetParamsChanged();
    }
}

void Application::render() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    if (m_renderer) {
        m_renderer->render();
    }
    
    if (m_uiController) {
        m_uiController->render();
    }
    
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Application::initializeCUDA() {
    // Try to initialize CUDA
    if (CUDAManager::getInstance().initialize()) {
        m_simulation = std::make_unique<CUDASimulationEngine>();
        m_useCUDA = true;
        LOG_INFO("CUDA initialized successfully");
        LOG_INFO("Using GPU: " + CUDAManager::getInstance().getDeviceName());
    } else {
        m_useCUDA = false;
        LOG_WARNING("CUDA initialization failed, using CPU simulation");
    }
}

void Application::initializeSimulation() {
    const auto& params = m_uiController->getParams();
    
    if (m_useCUDA && m_simulation) {
        // Initialize CUDA simulation
        m_simulation->initialize(params.rodPoints, params.rodLength);
        m_simulation->setMaterialProperties(params.thermalConductivity, 
                                           params.density, 
                                           params.specificHeat);
        m_simulation->setTemperatureRange(params.minTemp, params.maxTemp);
        m_simulation->setColorScheme(params.colorScheme);
        m_simulation->setHeatSourceTemp(params.heatSourceTemp);
        m_simulation->setAmbientTemp(params.ambientTemp);
        m_simulation->reset();
        
        // Get initial temperatures for rendering
        m_temperatures = m_simulation->getTemperatures();
    } else {
        // CPU fallback
        m_temperatures.resize(params.rodPoints);
        for (int i = 0; i < params.rodPoints; ++i) {
            if (i == 0) {
                m_temperatures[i] = params.heatSourceTemp;
            } else {
                m_temperatures[i] = params.ambientTemp;
            }
        }
    }
    
    // Update renderer with initial data
    m_renderer->setRodData(m_temperatures);
    m_simulationTime = 0.0f;
    
    LOG_INFO("Simulation initialized with " + std::to_string(params.rodPoints) + " points " +
             (m_useCUDA ? "(CUDA)" : "(CPU)"));
}

void Application::updateSimulation(float deltaTime) {
    const auto& params = m_uiController->getParams();
    
    if (!params.isPaused) {
        if (m_useCUDA && m_simulation) {
            // Update CUDA simulation
            m_simulation->setPaused(false);
            m_simulation->update(deltaTime);
            m_temperatures = m_simulation->getTemperatures();
            m_simulationTime = m_simulation->getSimulationTime();
        } else {
            // CPU fallback simulation
            float alpha = params.thermalConductivity / (params.density * params.specificHeat);
            float dx = params.rodLength / params.rodPoints;
            float dt = params.autoTimeStep ? 
                       std::min(0.5f * dx * dx / alpha, 0.01f) : 
                       params.timeStep;
            
            std::vector<float> newTemps = m_temperatures;
            for (int i = 1; i < params.rodPoints - 1; ++i) {
                float laplacian = (m_temperatures[i+1] - 2*m_temperatures[i] + m_temperatures[i-1]) / (dx * dx);
                newTemps[i] = m_temperatures[i] + alpha * dt * laplacian;
            }
            
            newTemps[0] = params.heatSourceTemp;
            newTemps[params.rodPoints - 1] = params.ambientTemp;
            
            m_temperatures = newTemps;
            m_simulationTime += dt;
        }
        
        // Update renderer
        m_renderer->setRodData(m_temperatures);
    } else if (m_useCUDA && m_simulation) {
        m_simulation->setPaused(true);
    }
}

void Application::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (app) {
        app->m_width = width;
        app->m_height = height;
        glViewport(0, 0, width, height);
        if (app->m_renderer) {
            app->m_renderer->resize(width, height);
        }
    }
}

void Application::errorCallback(int error, const char* description) {
    LOG_ERROR("GLFW Error (" + std::to_string(error) + "): " + std::string(description));
}

void Application::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (!app || action != GLFW_PRESS) return;
    
    switch (key) {
        case GLFW_KEY_SPACE:
            if (app->m_uiController) {
                app->m_uiController->togglePause();
            }
            break;
        case GLFW_KEY_R:
            if (app->m_uiController) {
                app->m_uiController->reset();
                app->initializeSimulation();
            }
            break;
    }
}