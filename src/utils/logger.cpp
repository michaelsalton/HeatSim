#include "logger.h"

void Logger::enableFileLogging(const std::string& filename) {
    if (m_logFile.is_open()) {
        m_logFile.close();
    }
    m_logFile.open(filename, std::ios::app);
    m_fileLogging = m_logFile.is_open();
}

void Logger::disableFileLogging() {
    if (m_logFile.is_open()) {
        m_logFile.close();
    }
    m_fileLogging = false;
}

void Logger::log(LogLevel level, const std::string& message, const std::string& file, int line) {
    if (level < m_minLevel) return;
    
    std::stringstream ss;
    ss << "[" << getCurrentTime() << "] ";
    ss << "[" << levelToString(level) << "] ";
    
    if (!file.empty() && line > 0) {
        size_t lastSlash = file.find_last_of("/\\");
        std::string filename = (lastSlash != std::string::npos) ? file.substr(lastSlash + 1) : file;
        ss << "[" << filename << ":" << line << "] ";
    }
    
    ss << message;
    
    std::string logMessage = ss.str();
    
    // Console output with colors
    switch (level) {
        case LogLevel::DEBUG:
            std::cout << "\033[36m" << logMessage << "\033[0m" << std::endl;
            break;
        case LogLevel::INFO:
            std::cout << "\033[32m" << logMessage << "\033[0m" << std::endl;
            break;
        case LogLevel::WARNING:
            std::cout << "\033[33m" << logMessage << "\033[0m" << std::endl;
            break;
        case LogLevel::ERROR:
            std::cerr << "\033[31m" << logMessage << "\033[0m" << std::endl;
            break;
        case LogLevel::CRITICAL:
            std::cerr << "\033[35m" << logMessage << "\033[0m" << std::endl;
            break;
    }
    
    // File output
    if (m_fileLogging && m_logFile.is_open()) {
        m_logFile << logMessage << std::endl;
        m_logFile.flush();
    }
}

std::string Logger::getCurrentTime() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

std::string Logger::levelToString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::CRITICAL: return "CRITICAL";
        default: return "UNKNOWN";
    }
}