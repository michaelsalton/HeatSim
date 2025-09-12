#pragma once

#include <string>
#include <memory>
#include <vector>
#include <GLFW/glfw3.h>

class Renderer;
class UIController;

class Application {
public:
    Application(const std::string& title, int width, int height);
    ~Application();
    
    void run();
    
private:
    void init();
    void cleanup();
    void processInput();
    void update(float deltaTime);
    void render();
    void initializeSimulation();
    void updateSimulation(float deltaTime);
    
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void errorCallback(int error, const char* description);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    
    GLFWwindow* m_window;
    std::unique_ptr<Renderer> m_renderer;
    std::unique_ptr<UIController> m_uiController;
    
    std::string m_title;
    int m_width;
    int m_height;
    bool m_running;
    
    float m_lastFrameTime;
    
    // Simulation data
    std::vector<float> m_temperatures;
    float m_simulationTime;
};