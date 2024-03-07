/******************************************************************************
 * GPUPrefixSums
 * Chained Scan with Decoupled Lookback Implementation
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 3/5/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 * Based off of Research by:
 *          Duane Merrill, Nvidia Corporation
 *          Michael Garland, Nvidia Corporation
 *          https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back
 * 
 ******************************************************************************/
#include "ChainedScanDecoupledLookback.cuh"

#define PART_VEC_SIZE	768
#define GRID_DIM        256

#define WARP_PARTITIONS 3
#define WARP_PART_SIZE  96
#define WARP_PART_START (WARP_INDEX * WARP_PART_SIZE)
#define PART_START      (partitionIndex * PART_VEC_SIZE)

#define FLAG_NOT_READY  0           //Flag indicating this partition tile's local reduction is not ready
#define FLAG_REDUCTION  1           //Flag indicating this partition tile's local reduction is ready
#define FLAG_INCLUSIVE  2           //Flag indicating this partition tile has summed all preceding tiles and added to its sum.
#define FLAG_MASK       3           //Mask used to retrieve the flag
 
 //Atomically acquire partition index
__device__ __forceinline__ uint32_t AcquirePartitionIndex(
    uint32_t& _broadcast,
    volatile uint32_t* _index)
{
    if (threadIdx.x == 0)
        _broadcast = atomicAdd((uint32_t*)&_index[0], 1);
}

__device__ __forceinline__ void LocalReduceDeviceBroadcast(
    uint32_t* _reduction,
    volatile uint32_t* _threadBlockReduction,
    const uint32_t& _partIndex)
{
    if (threadIdx.x < GRID_DIM / LANE_COUNT)
        _reduction[threadIdx.x] = ActiveInclusiveWarpScan(_reduction[threadIdx.x]);

    if (threadIdx.x == GRID_DIM / LANE_COUNT - 1)
    {
        atomicAdd((uint32_t*)&_threadBlockReduction[_partIndex],
            (_partIndex ? FLAG_REDUCTION : FLAG_INCLUSIVE) | _reduction[threadIdx.x] << 2);
    }
}

//lookback, non-divergent single warp
__device__ __forceinline__ void Lookback(
    uint32_t& _broadcast,
    volatile uint32_t* _threadBlockReduction,
    const uint32_t& _partIndex,
    uint32_t& _prevReduction)
{
    uint32_t k = _partIndex + LANE_COUNT - getLaneId();

    while (true)
    {
        const uint32_t flagPayload = k > LANE_COUNT ? _threadBlockReduction[k - LANE_COUNT - 1] : FLAG_INCLUSIVE;

        if (__all_sync(0xffffffff, (flagPayload & FLAG_MASK) > FLAG_NOT_READY))
        {
            uint32_t inclusiveBallot = __ballot_sync(0xffffffff, (flagPayload & FLAG_MASK) == FLAG_INCLUSIVE);
            if (inclusiveBallot)
            {
                _prevReduction += WarpReduceSum(getLaneId() < __ffs(inclusiveBallot) ? flagPayload >> 2 : 0);

                if (getLaneId() == 0)
                {
                    _broadcast = _prevReduction;
                    atomicAdd((uint32_t*)&_threadBlockReduction[_partIndex],
                        1 | (_prevReduction << 2));
                }
                break;
            }
            else
            {
                _prevReduction += WarpReduceSum(flagPayload >> 2);
                k -= LANE_COUNT;
            }
        }
    }
}

__global__ void ChainedScanDecoupledLookback::CSDLExclusive(
	uint32_t* scan,
    volatile uint32_t* threadBlockReduction,
	volatile uint32_t* index,
	uint32_t alignedSize)
{
    __shared__ uint4 s_csdl[PART_VEC_SIZE];
    __shared__ uint32_t s_reduction[GRID_DIM / LANE_COUNT];
    __shared__ uint32_t s_broadcast;

    AcquirePartitionIndex(s_broadcast, index);
    __syncthreads();
    const uint32_t partitionIndex = s_broadcast;

    if (partitionIndex < gridDim.x - 1)
    {
        ScanExclusiveFull(
            scan,
            s_reduction,
            s_csdl,
            PART_START,
            WARP_PARTITIONS,
            WARP_PART_START);
    }

    if (partitionIndex == gridDim.x - 1)
    {
        ScanExclusivePartial(
            scan,
            s_reduction,
            s_csdl,
            PART_START,
            WARP_PARTITIONS,
            WARP_PART_START,
            alignedSize);
    }
    __syncthreads();

    LocalReduceDeviceBroadcast(s_reduction, threadBlockReduction, partitionIndex);
        
    uint32_t prevReduction = 0;
    if (partitionIndex && threadIdx.x < LANE_COUNT)
        Lookback(s_broadcast, threadBlockReduction, partitionIndex, prevReduction);
    __syncthreads();
    
    if (threadIdx.x >= LANE_COUNT)
        prevReduction += s_broadcast + s_reduction[WARP_INDEX - 1];

    if (partitionIndex < gridDim.x - 1)
    {
        DownSweepFull(
            scan,
            s_csdl,
            prevReduction,
            PART_START,
            WARP_PARTITIONS,
            WARP_PART_START);
    }


    if (partitionIndex == gridDim.x - 1)
    {
        DownSweepPartial(
            scan,
            s_csdl,
            prevReduction,
            PART_START,
            WARP_PARTITIONS,
            WARP_PART_START,
            alignedSize);
    }
}

__global__ void ChainedScanDecoupledLookback::CSDLInclusive(
	uint32_t* scan,
    volatile uint32_t* threadBlockReduction,
	volatile uint32_t* index,
	uint32_t alignedSize)
{
    __shared__ uint4 s_csdl[PART_VEC_SIZE];
    __shared__ uint32_t s_reduction[GRID_DIM / LANE_COUNT];
    __shared__ uint32_t s_broadcast;

    AcquirePartitionIndex(s_broadcast, index);
    __syncthreads();
    const uint32_t partitionIndex = s_broadcast;

    if (partitionIndex < gridDim.x - 1)
    {
        ScanInclusiveFull(
            scan,
            s_reduction,
            s_csdl,
            PART_START,
            WARP_PARTITIONS,
            WARP_PART_START);
    }

    if (partitionIndex == gridDim.x - 1)
    {
        ScanInclusivePartial(
            scan,
            s_reduction,
            s_csdl,
            PART_START,
            WARP_PARTITIONS,
            WARP_PART_START,
            alignedSize);
    }
    __syncthreads();

    LocalReduceDeviceBroadcast(s_reduction, threadBlockReduction, partitionIndex);

    uint32_t prevReduction = 0;
    if (partitionIndex && threadIdx.x < LANE_COUNT)
        Lookback(s_broadcast, threadBlockReduction, partitionIndex, prevReduction);
    __syncthreads();

    if (threadIdx.x >= LANE_COUNT)
        prevReduction += s_broadcast + s_reduction[WARP_INDEX - 1];

    if (partitionIndex < gridDim.x - 1)
    {
        DownSweepFull(
            scan,
            s_csdl,
            prevReduction,
            PART_START,
            WARP_PARTITIONS,
            WARP_PART_START);
    }
        

    if (partitionIndex == gridDim.x - 1)
    {
        DownSweepPartial(
            scan,
            s_csdl,
            prevReduction,
            PART_START,
            WARP_PARTITIONS,
            WARP_PART_START,
            alignedSize);
    }
}