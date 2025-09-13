#include "heat_kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

// Shared memory tile size
#define TILE_SIZE 256

/**
 * Heat Diffusion Kernel - Implements 1D heat equation using explicit finite difference
 * 
 * Physics:
 * The 1D heat equation: ∂T/∂t = α ∂²T/∂x²
 * where α = k/(ρc) is the thermal diffusivity
 * 
 * Discretization:
 * Time derivative: (T[i]^(n+1) - T[i]^n) / Δt
 * Space derivative: (T[i+1]^n - 2*T[i]^n + T[i-1]^n) / Δx²
 * 
 * Update equation:
 * T[i]^(n+1) = T[i]^n + α*Δt/Δx² * (T[i+1]^n - 2*T[i]^n + T[i-1]^n)
 * 
 * Stability requirement (CFL condition):
 * Δt ≤ 0.5 * Δx² / α
 */
__global__ void heatDiffusionKernel(
    float* d_temperature,
    float* d_temperatureNew,
    float alpha,
    float dx,
    float dt,
    int numPoints) {
    
    extern __shared__ float tile[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory with boundary padding
    if (gid < numPoints) {
        tile[tid + 1] = d_temperature[gid];
    }
    
    // Load halo cells
    if (tid == 0 && gid > 0) {
        tile[0] = d_temperature[gid - 1];
    }
    if (tid == blockDim.x - 1 && gid < numPoints - 1) {
        tile[tid + 2] = d_temperature[gid + 1];
    }
    
    __syncthreads();
    
    // Compute new temperature using central difference (skip boundaries)
    if (gid > 0 && gid < numPoints - 1) {
        // Correct order: T[i-1] - 2*T[i] + T[i+1]
        float laplacian = (tile[tid] - 2.0f * tile[tid + 1] + tile[tid + 2]) / (dx * dx);
        d_temperatureNew[gid] = tile[tid + 1] + alpha * dt * laplacian;
    }
}

__global__ void applyBoundaryConditionsKernel(
    float* d_temperature,
    float heatSourceTemp,
    float ambientTemp,
    int numPoints) {
    
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Apply boundary conditions
    if (gid == 0) {
        d_temperature[0] = heatSourceTemp;
    }
    if (gid == numPoints - 1) {
        d_temperature[numPoints - 1] = ambientTemp;
    }
}

__global__ void temperatureToColorKernel(
    float* d_temperature,
    float* d_colorBuffer,
    float minTemp,
    float maxTemp,
    int colorScheme,
    int numPoints) {
    
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid >= numPoints) return;
    
    float temp = d_temperature[gid];
    float normalized = fminf(fmaxf((temp - minTemp) / (maxTemp - minTemp), 0.0f), 1.0f);
    
    float r, g, b;
    
    if (colorScheme == 0) {  // Heat map
        if (normalized < 0.25f) {
            float t = normalized * 4.0f;
            r = 0.0f;
            g = t;
            b = 1.0f;
        } else if (normalized < 0.5f) {
            float t = (normalized - 0.25f) * 4.0f;
            r = 0.0f;
            g = 1.0f;
            b = 1.0f - t;
        } else if (normalized < 0.75f) {
            float t = (normalized - 0.5f) * 4.0f;
            r = t;
            g = 1.0f;
            b = 0.0f;
        } else {
            float t = (normalized - 0.75f) * 4.0f;
            r = 1.0f;
            g = 1.0f - t;
            b = 0.0f;
        }
    } else if (colorScheme == 1) {  // Grayscale
        r = g = b = normalized;
    } else {  // Plasma
        r = sinf(normalized * 3.14159f);
        g = sinf(normalized * 3.14159f + 2.0f) * 0.7f;
        b = cosf(normalized * 3.14159f) * 0.8f;
    }
    
    // Store as RGB
    d_colorBuffer[gid * 3 + 0] = r;
    d_colorBuffer[gid * 3 + 1] = g;
    d_colorBuffer[gid * 3 + 2] = b;
}

__global__ void initializeTemperatureKernel(
    float* d_temperature,
    float initialTemp,
    int numPoints) {
    
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid < numPoints) {
        d_temperature[gid] = initialTemp;
    }
}

// Custom atomic functions for float min/max (not natively supported in all CUDA versions)
__device__ __forceinline__ float atomicMinFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                       __float_as_int(fminf(value, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                       __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// Kernel to compute statistics
__global__ void computeStatisticsKernel(
    float* d_temperature,
    float* d_minTemp,
    float* d_maxTemp,
    float* d_avgTemp,
    int numPoints) {
    
    extern __shared__ float sharedData[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    float localMin = 1e10f;
    float localMax = -1e10f;
    float localSum = 0.0f;
    
    // Load and compute local statistics
    if (gid < numPoints) {
        float temp = d_temperature[gid];
        localMin = temp;
        localMax = temp;
        localSum = temp;
    }
    
    sharedData[tid] = localMin;
    sharedData[tid + blockDim.x] = localMax;
    sharedData[tid + 2 * blockDim.x] = localSum;
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] = fminf(sharedData[tid], sharedData[tid + s]);
            sharedData[tid + blockDim.x] = fmaxf(sharedData[tid + blockDim.x], 
                                                  sharedData[tid + blockDim.x + s]);
            sharedData[tid + 2 * blockDim.x] += sharedData[tid + 2 * blockDim.x + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        atomicMinFloat(d_minTemp, sharedData[0]);
        atomicMaxFloat(d_maxTemp, sharedData[blockDim.x]);
        atomicAdd(d_avgTemp, sharedData[2 * blockDim.x]);
    }
}

// Host functions
void launchHeatDiffusionKernel(
    float* d_temperature,
    float* d_temperatureNew,
    float alpha,
    float dx,
    float dt,
    int numPoints) {
    
    int blockSize = 256;
    int gridSize = (numPoints + blockSize - 1) / blockSize;
    size_t sharedMemSize = (blockSize + 2) * sizeof(float);
    
    heatDiffusionKernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_temperature, d_temperatureNew, alpha, dx, dt, numPoints
    );
}

void launchBoundaryConditionsKernel(
    float* d_temperature,
    float heatSourceTemp,
    float ambientTemp,
    int numPoints) {
    
    int blockSize = 256;
    int gridSize = (numPoints + blockSize - 1) / blockSize;
    
    applyBoundaryConditionsKernel<<<gridSize, blockSize>>>(
        d_temperature, heatSourceTemp, ambientTemp, numPoints
    );
}

void launchTemperatureToColorKernel(
    float* d_temperature,
    float* d_colorBuffer,
    float minTemp,
    float maxTemp,
    int colorScheme,
    int numPoints) {
    
    int blockSize = 256;
    int gridSize = (numPoints + blockSize - 1) / blockSize;
    
    temperatureToColorKernel<<<gridSize, blockSize>>>(
        d_temperature, d_colorBuffer, minTemp, maxTemp, colorScheme, numPoints
    );
}

void launchInitializeTemperatureKernel(
    float* d_temperature,
    float initialTemp,
    int numPoints) {
    
    int blockSize = 256;
    int gridSize = (numPoints + blockSize - 1) / blockSize;
    
    initializeTemperatureKernel<<<gridSize, blockSize>>>(
        d_temperature, initialTemp, numPoints
    );
}