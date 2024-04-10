/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 3/5/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#pragma once
#include <stdio.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//General macros
#define LANE_COUNT          32							//Threads in a warp
#define LANE_MASK           31							//Mask of the lane count
#define LANE_LOG            5							//log2(LANE_COUNT)
#define WARP_INDEX          (threadIdx.x >> LANE_LOG)	//Warp of a thread

//PTX functions
__device__ __forceinline__ uint32_t getLaneId() 
{
    uint32_t laneId;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    return laneId;
}

__device__ __forceinline__ unsigned getLaneMaskLt() 
{
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
    return mask;
}

__device__ __forceinline__ uint32_t CircularShift(uint32_t mask, uint32_t val)
{
    return __shfl_sync(mask, val, getLaneId() + LANE_MASK & LANE_MASK);
}

//Warp scans
__device__ __forceinline__ uint32_t InclusiveWarpScan(uint32_t val)
{
    #pragma unroll
    for (int i = 1; i <= 16; i <<= 1) // 16 = LANE_COUNT >> 1
    {
        const uint32_t t = __shfl_up_sync(0xffffffff, val, i, LANE_COUNT);
        if (getLaneId() >= i) val += t;
    }

    return val;
}

__device__ __forceinline__ uint32_t ActiveInclusiveWarpScan(uint32_t val)
{
    const uint32_t mask = __activemask();
    #pragma unroll
    for (int i = 1; i <= 16; i <<= 1)
    {
        const uint32_t t = __shfl_up_sync(mask, val, i, LANE_COUNT);
        if (getLaneId() >= i) val += t;
    }

    return val;
}

__device__ __forceinline__ uint32_t InclusiveWarpScanCircularShift(uint32_t val)
{
    #pragma unroll
    for (int i = 1; i <= 16; i <<= 1) // 16 = LANE_COUNT >> 1
    {
        const uint32_t t = __shfl_up_sync(0xffffffff, val, i, LANE_COUNT);
        if (getLaneId() >= i) val += t;
    }

    return CircularShift(0xffffffff, val);
}

__device__ __forceinline__ uint32_t WarpReduceSum(uint32_t val)
{
    #pragma unroll
    for (int mask = 16; mask; mask >>= 1) // 16 = LANE_COUNT >> 1
        val += __shfl_xor_sync(0xffffffff, val, mask, LANE_COUNT);

    return val;
}

__device__ __forceinline__ uint32_t ActiveWarpReduceSum(uint32_t val)
{
    uint32_t activeMask = __activemask();

    #pragma unroll
    for (int mask = 16; mask; mask >>= 1) // 16 = LANE_COUNT >> 1
        val += __shfl_xor_sync(activeMask, val, mask, LANE_COUNT);

    return val;
}

__device__ __forceinline__ uint4 SetXAddYZW(uint32_t valToAdd, uint4 val)
{
    return make_uint4(valToAdd, val.y + valToAdd, val.z + valToAdd, val.w + valToAdd);
}

__device__ __forceinline__ uint4 AddUintToUint4(uint32_t valToAdd, uint4 val)
{
    return make_uint4(val.x + valToAdd, val.y + valToAdd, val.z + valToAdd, val.w + valToAdd);
}

__device__ __forceinline__ uint32_t ReduceUint4(uint4 val)
{
    return val.x + val.y + val.z + val.w;
}