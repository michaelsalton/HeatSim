#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <iomanip>

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    CRITICAL = 4
};

class Logger {
public:
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }
    
    void setLogLevel(LogLevel level) { m_minLevel = level; }
    void enableFileLogging(const std::string& filename);
    void disableFileLogging();
    
    void log(LogLevel level, const std::string& message, const std::string& file = "", int line = 0);
    
    void debug(const std::string& msg) { log(LogLevel::DEBUG, msg); }
    void info(const std::string& msg) { log(LogLevel::INFO, msg); }
    void warning(const std::string& msg) { log(LogLevel::WARNING, msg); }
    void error(const std::string& msg) { log(LogLevel::ERROR, msg); }
    void critical(const std::string& msg) { log(LogLevel::CRITICAL, msg); }
    
private:
    Logger() : m_minLevel(LogLevel::INFO), m_fileLogging(false) {}
    ~Logger() { if (m_logFile.is_open()) m_logFile.close(); }
    
    std::string getCurrentTime();
    std::string levelToString(LogLevel level);
    
    LogLevel m_minLevel;
    bool m_fileLogging;
    std::ofstream m_logFile;
};

#define LOG_DEBUG(msg) Logger::getInstance().log(LogLevel::DEBUG, msg, __FILE__, __LINE__)
#define LOG_INFO(msg) Logger::getInstance().log(LogLevel::INFO, msg, __FILE__, __LINE__)
#define LOG_WARNING(msg) Logger::getInstance().log(LogLevel::WARNING, msg, __FILE__, __LINE__)
#define LOG_ERROR(msg) Logger::getInstance().log(LogLevel::ERROR, msg, __FILE__, __LINE__)
#define LOG_CRITICAL(msg) Logger::getInstance().log(LogLevel::CRITICAL, msg, __FILE__, __LINE__)