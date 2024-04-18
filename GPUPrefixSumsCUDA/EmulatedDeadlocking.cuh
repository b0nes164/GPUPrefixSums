/******************************************************************************
 * GPUPrefixSums
 * Emulated Deadlocking
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 4/15/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#pragma once
#include "Utils.cuh"
#include "ScanCommon.cuh"

namespace EmulatedDeadlocking
{
    __global__ void EmulatedDeadlockSpinning(
        uint32_t* scan,
        volatile uint32_t* threadBlockReduction,
        volatile uint32_t* index,
        uint32_t vectorizedSize,
        uint32_t partMask,
        uint32_t maxSpinCount);
}