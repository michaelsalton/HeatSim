#include <iostream>
#include <memory>
#include "core/application.h"

int main() {
    try {
        auto app = std::make_unique<Application>("Heat Sim", 1920, 1080);
        app->run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}