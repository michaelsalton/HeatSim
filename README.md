# Heat Sim

A real-time 1D heat transfer visualization built with C++, CUDA, and ImGui.

## Features

- Real-time 1D heat diffusion simulation
- Interactive heat source temperature control
- Variable rod length and material properties
- Color-mapped temperature visualization
- Play/pause/reset simulation controls
- GPU-accelerated calculations with CUDA

## Requirements

### System Requirements
- Linux/Windows with NVIDIA GPU (Compute Capability 3.5+)
- OpenGL 4.3+ support
- CUDA 11.0+ runtime (optional, for GPU acceleration)

### Build Dependencies
- CMake 3.18+
- C++17 compatible compiler (GCC 8+, Clang 9+, MSVC 2019+)
- GLFW3
- OpenGL development libraries
- CUDA Toolkit 11.0+ (optional)

## Installation

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install build-essential cmake
sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
sudo apt-get install libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
```

### Fedora/RHEL
```bash
sudo dnf install cmake gcc-c++ make
sudo dnf install glfw-devel mesa-libGL-devel mesa-libGLU-devel
sudo dnf install libXrandr-devel libXinerama-devel libXcursor-devel libXi-devel
```

### CUDA (Optional)
Download and install CUDA Toolkit from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)

## Building

### Using CMake
```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### Using Make (Alternative)
```bash
make
```

## Running

```bash
./build/bin/HeatSim
```

## Controls

- **ESC**: Exit application
- **ImGui Panel**: Interactive controls for simulation parameters

## Project Structure

```
HeatSim/
├── src/              # Source files
├── include/          # External headers
├── shaders/          # GLSL shaders
├── assets/           # Resources (fonts, textures)
├── docs/             # Documentation
├── build/            # Build output (generated)
└── CMakeLists.txt    # CMake configuration
```

## Documentation

See the [docs](docs/) folder for detailed documentation:
- [Project Overview](docs/heat_sim_project_overview.md)

## License

This project is for educational purposes.