/******************************************************************************
 * GPUPrefixSums
 * Reduce then Scan 
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 11/30/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#pragma once
#include "ScanCommon.cuh"
#include "Utils.cuh"

namespace ReduceThenScan {
    template <uint32_t WARPS, uint32_t PER_THREAD>
    __global__ void Reduce(uint32_t* scan, uint32_t* threadBlockReductions,
                           const uint32_t vectorizedSize) {
        constexpr uint32_t PART_VEC_SIZE = WARPS * LANE_COUNT * PER_THREAD;
        __shared__ uint32_t s_red[WARPS];

        uint32_t warpReduction = 0;
        //full
        if (blockIdx.x < gridDim.x - 1) {
            #pragma unroll
            for (uint32_t i = threadIdx.x + PART_VEC_SIZE * blockIdx.x, k = 0; k < PER_THREAD;
                 i += blockDim.x, ++k) {
                warpReduction += WarpReduceSum(ReduceUint4(reinterpret_cast<uint4*>(scan)[i]));
            }
        }

        //partial
        if (blockIdx.x == gridDim.x - 1) {
            #pragma unroll
            for (uint32_t i = threadIdx.x + PART_VEC_SIZE * blockIdx.x, k = 0; k < PER_THREAD;
                 i += blockDim.x, ++k) {
                warpReduction += WarpReduceSum(
                    i < vectorizedSize ? 0 : ReduceUint4(reinterpret_cast<uint4*>(scan)[i]));
            }
        }

        if (!getLaneId()) {
            s_red[WARP_INDEX] = warpReduction;
        }
        __syncthreads();

        if (threadIdx.x < LANE_COUNT) {
            const bool pred = threadIdx.x < WARPS;
            const uint32_t blockReduction = WarpReduceSum(pred ? s_red[threadIdx.x] : 0);
            if (!threadIdx.x) {
                threadBlockReductions[blockIdx.x] = blockReduction;
            }
        }
    }

    // Unvectorized
    template <uint32_t WARPS, uint32_t PER_THREAD>
    __global__ void RootScan(uint32_t* threadBlockReductions, const uint32_t threadBlocks) {
        constexpr uint32_t PART_SIZE = WARPS * PER_THREAD * LANE_COUNT;
        __shared__ uint32_t s_red[WARPS];

        const uint32_t alignedSize = (threadBlocks + PART_SIZE - 1) / PART_SIZE * PART_SIZE;
        const uint32_t threadStart = getLaneId() + WARP_INDEX * PER_THREAD * LANE_COUNT;

        uint32_t prevReduction = 0;
        uint32_t tScan[PER_THREAD];
        for (uint32_t i = 0; i < alignedSize; i += PART_SIZE) {
            uint32_t prev = 0;
            #pragma unroll
            for (uint32_t j = threadStart + i, k = 0; k < PER_THREAD; j += LANE_COUNT, ++k) {
                if (j < threadBlocks) {
                    tScan[k] = threadBlockReductions[j];
                }
                tScan[k] = InclusiveWarpScan(tScan[k]) + prev;
                prev = __shfl_sync(0xffffffff, tScan[k], LANE_MASK);
            }

            if (!getLaneId()) {
                s_red[WARP_INDEX] = prev;
            }
            __syncthreads();

            if (threadIdx.x < LANE_COUNT) {
                const bool pred = threadIdx.x < WARPS;
                const uint32_t t = InclusiveWarpScan(pred ? s_red[threadIdx.x] : 0);
                if (pred) {
                    s_red[threadIdx.x] = t;
                }
            }
            __syncthreads();

            const uint32_t combinedReduction =
                (threadIdx.x >= LANE_COUNT ? s_red[WARPS - 1] : 0) + prevReduction;

            #pragma unroll
            for (uint32_t j = threadStart + i, k = 0; k < PER_THREAD; j += LANE_COUNT, ++k) {
                if (j < threadBlocks) {
                    threadBlockReductions[j] = tScan[k] + combinedReduction;
                }
            }

            prevReduction += s_red[WARPS - 1];
            __syncthreads();
        }
    }

    template <uint32_t WARPS, uint32_t PER_THREAD>
    __global__ void DownSweepInclusive(uint32_t* scan, uint32_t* threadBlockReductions,
                                       const uint32_t vectorizedSize) {
        constexpr uint32_t PART_VEC_SIZE = WARPS * LANE_COUNT * PER_THREAD;
        __shared__ uint32_t s_warpReduction[WARPS];

        uint4 tScan[PER_THREAD];
        const uint32_t offset = WARP_INDEX * LANE_COUNT * PER_THREAD + blockIdx.x * PART_VEC_SIZE;
        if (blockIdx.x < gridDim.x - 1) {
            ScanInclusiveFull<PER_THREAD>(tScan, scan, s_warpReduction, offset);
        }

        if (blockIdx.x == gridDim.x - 1) {
            ScanInclusivePartial<PER_THREAD>(tScan, scan, s_warpReduction, offset, vectorizedSize);
        }
        __syncthreads();

        if (threadIdx.x < LANE_COUNT) {
            const bool pred = threadIdx.x < WARPS;
            const uint32_t t = InclusiveWarpScan(pred ? s_warpReduction[threadIdx.x] : 0);
            if (pred) {
                s_warpReduction[threadIdx.x] = t;
            }
        }
        __syncthreads();

        const uint32_t prevReduction =
            (blockIdx.x ? threadBlockReductions[blockIdx.x - 1] : 0) +
            (threadIdx.x >= LANE_COUNT ? s_warpReduction[WARP_INDEX - 1] : 0);

        if (blockIdx.x < gridDim.x - 1) {
            PropagateFull<PER_THREAD>(tScan, scan, prevReduction, offset);
        }

        if (blockIdx.x == gridDim.x - 1) {
            PropagatePartial<PER_THREAD>(tScan, scan, prevReduction, offset, vectorizedSize);
        }
    }

    template <uint32_t WARPS, uint32_t PER_THREAD>
    __global__ void DownSweepExclusive(uint32_t* scan, uint32_t* threadBlockReductions,
                                       const uint32_t vectorizedSize) {
        constexpr uint32_t PART_VEC_SIZE = WARPS * LANE_COUNT * PER_THREAD;
        __shared__ uint32_t s_warpReduction[WARPS];

        uint4 tScan[PER_THREAD];
        const uint32_t offset = WARP_INDEX * LANE_COUNT * PER_THREAD + blockIdx.x * PART_VEC_SIZE;
        if (blockIdx.x < gridDim.x - 1) {
            ScanExclusiveFull<PER_THREAD>(tScan, scan, s_warpReduction, offset);
        }

        if (blockIdx.x == gridDim.x - 1) {
            ScanExclusivePartial<PER_THREAD>(tScan, scan, s_warpReduction, offset, vectorizedSize);
        }
        __syncthreads();

        if (threadIdx.x < LANE_COUNT) {
            const bool pred = threadIdx.x < WARPS;
            const uint32_t t = InclusiveWarpScan(pred ? s_warpReduction[threadIdx.x] : 0);
            if (pred) {
                s_warpReduction[threadIdx.x] = t;
            }
        }
        __syncthreads();

        const uint32_t prevReduction =
            (blockIdx.x ? threadBlockReductions[blockIdx.x - 1] : 0) +
            (threadIdx.x >= LANE_COUNT ? s_warpReduction[WARP_INDEX - 1] : 0);

        if (blockIdx.x < gridDim.x - 1) {
            PropagateFull<PER_THREAD>(tScan, scan, prevReduction, offset);
        }

        if (blockIdx.x == gridDim.x - 1) {
            PropagatePartial<PER_THREAD>(tScan, scan, prevReduction, offset, vectorizedSize);
        }
    }
}  // namespace ReduceThenScan