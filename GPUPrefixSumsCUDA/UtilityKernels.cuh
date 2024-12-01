/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 11/30/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#pragma once
#include <stdint.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void InitOne(uint32_t* scan, uint32_t size) {
    const uint32_t increment = blockDim.x * gridDim.x;
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += increment) {
        scan[i] = 1;
    }
}

__global__ void ValidateInclusive(uint32_t* scan, uint32_t* errCount, uint32_t size) {
    const uint32_t increment = blockDim.x * gridDim.x;
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += increment) {
        if (scan[i] != i + 1) {
            atomicAdd(&errCount[0], 1);
        }
    }
}

__global__ void ValidateExclusive(uint32_t* scan, uint32_t* errCount, uint32_t size) {
    const uint32_t increment = blockDim.x * gridDim.x;
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += increment) {
        if (scan[i] != i) {
            atomicAdd(&errCount[0], 1);
        }
    }
}

__global__ void Print(uint32_t* toPrint, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        printf("%u: %u\n", i, toPrint[i]);
    }
}