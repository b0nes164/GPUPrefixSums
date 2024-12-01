/******************************************************************************
 * GPUPrefixSums
 * Emulated Deadlocking
 * Emulates the behaviour of a device without forward progress guarantees.
 * 
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 11/30/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#pragma once
#include "ScanCommon.cuh"
#include "Utils.cuh"

#define MAX_SPIN_COUNT 4
#define DEADLOCK_MASK 7
#define LOCKED 1
#define UNLOCKED 0

#define FLAG_NOT_READY 0
#define FLAG_REDUCTION 1
#define FLAG_INCLUSIVE 2
#define FLAG_MASK 3

namespace EmulatedDeadlocking {
    template <uint32_t WARPS, uint32_t PER_THREAD>
    __global__ void EmulatedDeadlock(uint32_t* scan, volatile uint32_t* threadBlockReduction,
                                     volatile uint32_t* bump, const uint32_t vectorizedSize) {
        constexpr uint32_t PART_VEC_SIZE = WARPS * LANE_COUNT * PER_THREAD;
        __shared__ uint32_t s_fallbackRed[WARPS];
        __shared__ uint32_t s_warpReduction[WARPS];
        __shared__ uint32_t s_lock;
        __shared__ uint32_t s_broadcast;

        // Atomically acquire partition index
        if (!threadIdx.x) {
            s_broadcast = atomicAdd((uint32_t*)&bump[0], 1);
            s_lock = LOCKED;
        }
        __syncthreads();
        const uint32_t partitionIndex = s_broadcast;

        uint4 tScan[PER_THREAD];
        const uint32_t offset =
            WARP_INDEX * LANE_COUNT * PER_THREAD + partitionIndex * PART_VEC_SIZE;
        if (partitionIndex < gridDim.x - 1) {
            ScanInclusiveFull<PER_THREAD>(tScan, scan, s_warpReduction, offset);
        }

        if (partitionIndex == gridDim.x - 1) {
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

        if (!threadIdx.x) {
            if (!(partitionIndex & DEADLOCK_MASK) && gridDim.x > 1) {
                while (!threadBlockReduction[partitionIndex]) {}
            }
            atomicExch((uint32_t*)&threadBlockReduction[partitionIndex],
                       (partitionIndex ? FLAG_REDUCTION : FLAG_INCLUSIVE) |
                           s_warpReduction[WARPS - 1] << 2);
        }
        __syncthreads();    //Purely for emulation
        __threadfence();

        // Lookback and potentially fallback, using a single threaded lookback
        if (partitionIndex) {
            uint32_t prevReduction = 0;
            uint32_t lookbackIndex = partitionIndex - 1;
            while (s_lock == LOCKED) {
                __syncthreads();
                if (!threadIdx.x) {
                    uint32_t spin = 0;
                    while (spin < MAX_SPIN_COUNT) {
                        const uint32_t flagPayload = threadBlockReduction[lookbackIndex];
                        if ((flagPayload & FLAG_MASK) > FLAG_NOT_READY) {
                            prevReduction += flagPayload >> 2;
                            spin = 0;
                            if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE) {
                                s_lock = UNLOCKED;
                                s_broadcast = prevReduction;
                                atomicExch((uint32_t*)&threadBlockReduction[partitionIndex],
                                           FLAG_INCLUSIVE |
                                               prevReduction + s_warpReduction[WARPS - 1] << 2);
                                break;
                            } else {
                                lookbackIndex--;
                            }
                        } else {
                            spin++;
                        }
                    }

                    // If the loop exited when spin == MAX_SPIN_COUNT,
                    // then the lookback failed, and a fallback must be initiated
                    // so we broadcast the fallback partition index for the whole threadblock to consume
                    if (spin == MAX_SPIN_COUNT) {
                        s_broadcast = lookbackIndex;
                    }
                }
                __syncthreads();

                // If still locked, fallback
                if (s_lock == LOCKED) {
                    const uint32_t fallbackIndex = s_broadcast;
                    uint32_t warpReduction = 0;
                    #pragma unroll
                    for (uint32_t i = threadIdx.x + PART_VEC_SIZE * fallbackIndex, k = 0;
                         k < PER_THREAD; i += blockDim.x, ++k) {
                        warpReduction +=
                            WarpReduceSum(ReduceUint4(reinterpret_cast<uint4*>(scan)[i]));
                    }

                    if (!getLaneId()) {
                        s_fallbackRed[WARP_INDEX] = warpReduction;
                    }
                    __syncthreads();

                    uint32_t fallbackReduction;
                    if (threadIdx.x < LANE_COUNT) {
                        const bool pred = threadIdx.x < WARPS;
                        fallbackReduction = WarpReduceSum(pred ? s_fallbackRed[threadIdx.x] : 0);
                    }
                    __syncthreads();

                    if (!threadIdx.x) {
                        const uint32_t fallbackPayload =
                            (fallbackIndex ? FLAG_REDUCTION : FLAG_INCLUSIVE) | fallbackReduction
                                                                                    << 2;
                        const uint32_t prev = atomicCAS(
                            (uint32_t*)&threadBlockReduction[fallbackIndex], 0, fallbackPayload);
                        prevReduction += prev ? (prev >> 2) : fallbackReduction;
                        if ((prev & FLAG_MASK) == FLAG_INCLUSIVE || !fallbackIndex) {
                            s_lock = UNLOCKED;
                            s_broadcast = prevReduction;
                            atomicExch(
                                (uint32_t*)&threadBlockReduction[partitionIndex],
                                FLAG_INCLUSIVE | prevReduction + s_warpReduction[WARPS - 1] << 2);
                        } else {
                            lookbackIndex--;
                        }
                    }
                    __syncthreads();
                }
            }
        }
        __threadfence();

        const uint32_t prevReduction =
            s_broadcast + (threadIdx.x >= LANE_COUNT ? s_warpReduction[WARP_INDEX - 1] : 0);

        if (partitionIndex < gridDim.x - 1) {
            PropagateFull<PER_THREAD>(tScan, scan, prevReduction, offset);
        }

        if (partitionIndex == gridDim.x - 1) {
            PropagatePartial<PER_THREAD>(tScan, scan, prevReduction, offset, vectorizedSize);
        }
    }
}  // namespace EmulatedDeadlocking

#undef MAX_SPIN_COUNT
#undef DEADLOCK_MASK
#undef LOCKED
#undef UNLOCKED

#undef FLAG_NOT_READY
#undef FLAG_REDUCTION
#undef FLAG_INCLUSIVE
#undef FLAG_MASK