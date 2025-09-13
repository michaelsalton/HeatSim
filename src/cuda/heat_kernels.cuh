#pragma once

#include <cuda_runtime.h>

// Heat diffusion kernel - applies finite difference method
void launchHeatDiffusionKernel(
    float* d_temperature,
    float* d_temperatureNew,
    float alpha,           // Thermal diffusivity
    float dx,              // Spatial step
    float dt,              // Time step
    int numPoints
);

// Boundary conditions kernel
void launchBoundaryConditionsKernel(
    float* d_temperature,
    float heatSourceTemp,
    float ambientTemp,
    int numPoints
);

// Temperature to color mapping kernel
void launchTemperatureToColorKernel(
    float* d_temperature,
    float* d_colorBuffer,
    float minTemp,
    float maxTemp,
    int colorScheme,
    int numPoints
);

// Initialize temperature array
void launchInitializeTemperatureKernel(
    float* d_temperature,
    float initialTemp,
    int numPoints
);

// Compute statistics (min, max, average)
void launchComputeStatisticsKernel(
    float* d_temperature,
    float* d_minTemp,
    float* d_maxTemp,
    float* d_avgTemp,
    int numPoints
);