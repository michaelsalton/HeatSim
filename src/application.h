#pragma once

#include <string>
#include <memory>
#include <GLFW/glfw3.h>

class Renderer;

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
    
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void errorCallback(int error, const char* description);
    
    GLFWwindow* m_window;
    std::unique_ptr<Renderer> m_renderer;
    
    std::string m_title;
    int m_width;
    int m_height;
    bool m_running;
    
    float m_lastFrameTime;
};