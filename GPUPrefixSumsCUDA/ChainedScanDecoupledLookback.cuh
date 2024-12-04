/******************************************************************************
 * GPUPrefixSums
 * Chained Scan with Decoupled Lookback Implementation
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 11/30/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 * Based off of Research by:
 *          Duane Merrill, Nvidia Corporation
 *          Michael Garland, Nvidia Corporation
 *          https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back
 *
 ******************************************************************************/
#pragma once
#include "ScanCommon.cuh"
#include "Utils.cuh"

#define FLAG_NOT_READY 0  // Flag indicating this partition tile's local reduction is not ready
#define FLAG_REDUCTION 1  // Flag indicating this partition tile's local reduction is ready
#define FLAG_INCLUSIVE \
    2  // Flag indicating this partition tile has summed all preceding tiles and added to its sum.
#define FLAG_MASK 3  // Mask used to retrieve the flag

namespace ChainedScanDecoupledLookback {
    // Two different ways to do the lookback, testing shows negligible performance difference:
    // lookback, single thread
    __device__ __forceinline__ void LookbackSingle(const uint32_t partIndex,
                                                   uint32_t localReduction, uint32_t& s_broadcast,
                                                   volatile uint32_t* threadBlockReduction) {
        uint32_t prevReduction = 0;
        uint32_t lookbackIndex = partIndex - 1;
        while (true) {
            const uint32_t flagPayload = threadBlockReduction[lookbackIndex];
            if ((flagPayload & FLAG_MASK) > FLAG_NOT_READY) {
                prevReduction += flagPayload >> 2;
                if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE) {
                    s_broadcast = prevReduction;
                    atomicExch((uint32_t*)&threadBlockReduction[partIndex],
                               FLAG_INCLUSIVE | prevReduction + localReduction << 2);
                    break;
                } else {
                    lookbackIndex--;
                }
            }
        }
    }

    // lookback, non-divergent single warp
    __device__ __forceinline__ void LookbackWarp(const uint32_t partIndex, uint32_t localReduction,
                                                 uint32_t& s_broadcast,
                                                 volatile uint32_t* threadBlockReduction) {
        uint32_t prevReduction = 0;
        uint32_t lookbackIndex = partIndex + LANE_COUNT - getLaneId();
        while (true) {
            const uint32_t flagPayload = lookbackIndex > LANE_COUNT
                                             ? threadBlockReduction[lookbackIndex - LANE_COUNT - 1]
                                             : FLAG_INCLUSIVE;
            if (__all_sync(0xffffffff, (flagPayload & FLAG_MASK) > FLAG_NOT_READY)) {
                uint32_t inclusiveBallot =
                    __ballot_sync(0xffffffff, (flagPayload & FLAG_MASK) == FLAG_INCLUSIVE);
                if (inclusiveBallot) {
                    prevReduction +=
                        WarpReduceSum(getLaneId() < __ffs(inclusiveBallot) ? flagPayload >> 2 : 0);
                    if (getLaneId() == 0) {
                        s_broadcast = prevReduction;
                        atomicExch((uint32_t*)&threadBlockReduction[partIndex],
                                   FLAG_INCLUSIVE | prevReduction + localReduction << 2);
                    }
                    break;
                } else {
                    prevReduction += WarpReduceSum(flagPayload >> 2);
                    lookbackIndex -= LANE_COUNT;
                }
            }
        }
    }

    template <uint32_t WARPS, uint32_t PER_THREAD>
    __device__ __forceinline__ void CSDL(
        uint32_t* scan, volatile uint32_t* threadBlockReduction, volatile uint32_t* bump,
        const uint32_t vectorizedSize,
        void (*ScanVariantFull)(uint4*, uint32_t*, uint32_t*, const uint32_t),
        void (*ScanVariantPartial)(uint4*, uint32_t*, uint32_t*, const uint32_t, const uint32_t)) {
        constexpr uint32_t PART_VEC_SIZE = WARPS * LANE_COUNT * PER_THREAD;
        __shared__ uint32_t s_warpReduction[WARPS];
        __shared__ uint32_t s_broadcast;

        // Atomically acquire partition index
        if (!threadIdx.x) {
            s_broadcast = atomicAdd((uint32_t*)&bump[0], 1);
        }
        __syncthreads();
        const uint32_t partitionIndex = s_broadcast;

        uint4 tScan[PER_THREAD];
        uint32_t offset = WARP_INDEX * LANE_COUNT * PER_THREAD + partitionIndex * PART_VEC_SIZE;
        if (partitionIndex < gridDim.x - 1) {
            (*ScanVariantFull)(tScan, scan, s_warpReduction, offset);
        }

        if (partitionIndex == gridDim.x - 1) {
            (*ScanVariantPartial)(tScan, scan, s_warpReduction, offset, vectorizedSize);
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

        if (!threadIdx.x) {
            atomicExch((uint32_t*)&threadBlockReduction[partitionIndex],
                       (partitionIndex ? FLAG_REDUCTION : FLAG_INCLUSIVE) |
                           s_warpReduction[WARPS - 1] << 2);
        }
        __threadfence();

        if (partitionIndex && threadIdx.x < LANE_COUNT) {
            LookbackWarp(partitionIndex, s_warpReduction[WARPS - 1], s_broadcast,
                         threadBlockReduction);
        }
        __syncthreads();

        const uint32_t prevReduction =
            s_broadcast + (threadIdx.x >= LANE_COUNT ? s_warpReduction[WARP_INDEX - 1] : 0);

        if (partitionIndex < gridDim.x - 1) {
            PropagateFull<PER_THREAD>(tScan, scan, prevReduction, offset);
        }

        if (partitionIndex == gridDim.x - 1) {
            PropagatePartial<PER_THREAD>(tScan, scan, prevReduction, offset, vectorizedSize);
        }
    }

    template <uint32_t WARPS, uint32_t PER_THREAD>
    __global__ void CSDLExclusive(uint32_t* scan, volatile uint32_t* threadBlockReduction,
                                  volatile uint32_t* bump, const uint32_t vectorizedSize) {
        CSDL<WARPS, PER_THREAD>(scan, threadBlockReduction, bump, vectorizedSize,
                                ScanExclusiveFull<PER_THREAD>, ScanExclusivePartial<PER_THREAD>);
    }

    template <uint32_t WARPS, uint32_t PER_THREAD>
    __global__ void CSDLInclusive(uint32_t* scan, volatile uint32_t* threadBlockReduction,
                                  volatile uint32_t* bump, const uint32_t vectorizedSize) {
        CSDL<WARPS, PER_THREAD>(scan, threadBlockReduction, bump, vectorizedSize,
                                ScanInclusiveFull<PER_THREAD>, ScanInclusivePartial<PER_THREAD>);
    }

}  // namespace ChainedScanDecoupledLookback

#undef FLAG_NOT_READY
#undef FLAG_REDUCTION
#undef FLAG_INCLUSIVE
#undef FLAG_MASK