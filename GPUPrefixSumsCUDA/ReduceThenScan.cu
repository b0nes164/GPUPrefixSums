/******************************************************************************
 * GPUPrefixSums
 * Reduce then Scan
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 3/5/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#include "ReduceThenScan.cuh"

#define PART_VEC_SIZE	768
#define BLOCK_DIM		256

#define WARP_PARTITIONS 3
#define WARP_PART_SIZE  96
#define WARP_PART_START (WARP_INDEX * WARP_PART_SIZE)
#define PART_START      (blockIdx.x * PART_VEC_SIZE)

__device__ __forceinline__ void LocalReduce(
    uint32_t* _reduction)
{
    if (threadIdx.x < BLOCK_DIM / LANE_COUNT)
        _reduction[threadIdx.x] = ActiveInclusiveWarpScan(_reduction[threadIdx.x]);
}

__global__ void ReduceThenScan::Reduce(
    uint32_t* scan,
    uint32_t* threadBlockReductions,
    uint32_t vectorizedSize)
{
    __shared__ uint32_t s_red[BLOCK_DIM / LANE_COUNT];
    //full
    if (blockIdx.x < gridDim.x - 1)
    {
        uint32_t warpReduction = 0;
        const uint32_t partEnd = (blockIdx.x + 1) * PART_VEC_SIZE;
        for (uint32_t i = threadIdx.x + PART_START; i < partEnd; i += blockDim.x)
            warpReduction += WarpReduceSum(ReduceUint4(reinterpret_cast<uint4*>(scan)[i]));

        if (!getLaneId())
            s_red[WARP_INDEX] = warpReduction;
    }

    //partial
    if (blockIdx.x == gridDim.x - 1)
    {
        uint32_t warpReduction = 0;
        const uint32_t partEnd = (blockIdx.x + 1) * PART_VEC_SIZE;
        for (uint32_t i = threadIdx.x + PART_START; i < partEnd; i += blockDim.x)
        {
            warpReduction += WarpReduceSum(
                i < vectorizedSize ? 0 :
                ReduceUint4(reinterpret_cast<uint4*>(scan)[i]));
        }

        if (!getLaneId())
            s_red[WARP_INDEX] = warpReduction;
    }
    __syncthreads();

    uint32_t blockReduction;
    if (threadIdx.x < BLOCK_DIM / LANE_COUNT)
        blockReduction = ActiveWarpReduceSum(s_red[WARP_INDEX]);

    if (!threadIdx.x)
        threadBlockReductions[blockIdx.x] = blockReduction;
}

__global__ void ReduceThenScan::Scan(
    uint32_t* threadBlockReductions,
    uint32_t threadBlocks)
{
    __shared__ uint32_t s_scan[BLOCK_DIM];

    uint32_t reduction = 0;
    const uint32_t circularLaneShift = getLaneId() + 1 & LANE_MASK;
    const uint32_t partitionsEnd = threadBlocks / blockDim.x * blockDim.x;

    uint32_t i = threadIdx.x;
    for (; i < partitionsEnd; i += blockDim.x)
    {
        s_scan[threadIdx.x] = threadBlockReductions[i];
        s_scan[threadIdx.x] = InclusiveWarpScan(s_scan[threadIdx.x]);
        __syncthreads();

        if (threadIdx.x < (blockDim.x >> LANE_LOG))
        {
            s_scan[(threadIdx.x + 1 << LANE_LOG) - 1] =
                ActiveInclusiveWarpScan(s_scan[(threadIdx.x + 1 << LANE_LOG) - 1]);
        }
        __syncthreads();

        threadBlockReductions[circularLaneShift + (i & ~LANE_MASK)] =
            (getLaneId() != LANE_MASK ? s_scan[threadIdx.x] : 0) +
            (threadIdx.x >= LANE_COUNT ? __shfl_sync(0xffffffff, s_scan[threadIdx.x - 1], 0) : 0) +
            reduction;

        reduction += s_scan[blockDim.x - 1];
        __syncthreads();
    }

    if (i < threadBlocks)
        s_scan[threadIdx.x] = threadBlockReductions[i];
    s_scan[threadIdx.x] = InclusiveWarpScan(s_scan[threadIdx.x]);
    __syncthreads();

    if (threadIdx.x < (blockDim.x >> LANE_LOG))
    {
        s_scan[(threadIdx.x + 1 << LANE_LOG) - 1] =
            ActiveInclusiveWarpScan(s_scan[(threadIdx.x + 1 << LANE_LOG) - 1]);
    }
    __syncthreads();

    const uint32_t index = circularLaneShift + (i & ~LANE_MASK);
    if (index < threadBlocks)
    {
        threadBlockReductions[index] =
            (getLaneId() != LANE_MASK ? s_scan[threadIdx.x] : 0) +
            (threadIdx.x >= LANE_COUNT ?
                s_scan[(threadIdx.x & ~LANE_MASK) - 1] : 0) +
            reduction;
    }
}

__global__ void ReduceThenScan::DownSweepExclusive(
    uint32_t* scan,
    uint32_t* threadBlockReductions,
    uint32_t vectorizedSize)
{
    __shared__ uint4 s_rts[PART_VEC_SIZE];
    __shared__ uint32_t s_reduction[BLOCK_DIM / LANE_COUNT];

    //full
    if (blockIdx.x < gridDim.x - 1)
    {
        ScanExclusiveFull(
            scan,
            s_reduction,
            s_rts,
            PART_START,
            WARP_PARTITIONS,
            WARP_PART_START);
    }

    //partial
    if (blockIdx.x == gridDim.x - 1)
    {
        ScanExclusivePartial(
            scan,
            s_reduction,
            s_rts,
            PART_START,
            WARP_PARTITIONS,
            WARP_PART_START,
            vectorizedSize);
    }

    uint32_t prevReduction = blockIdx.x ? threadBlockReductions[blockIdx.x] : 0;
    __syncthreads();

    LocalReduce(s_reduction);
    __syncthreads();

    if(threadIdx.x >= LANE_COUNT)
        prevReduction += s_reduction[WARP_INDEX - 1];

    if (blockIdx.x < gridDim.x - 1)
    {
        DownSweepFull(
            scan,
            s_rts,
            prevReduction,
            PART_START,
            WARP_PARTITIONS,
            WARP_PART_START);
    }


    if (blockIdx.x == gridDim.x - 1)
    {
        DownSweepPartial(
            scan,
            s_rts,
            prevReduction,
            PART_START,
            WARP_PARTITIONS,
            WARP_PART_START,
            vectorizedSize);
    }
}

__global__ void ReduceThenScan::DownSweepInclusive(
    uint32_t* scan,
    uint32_t* threadBlockReductions,
    uint32_t vectorizedSize)
{
    __shared__ uint4 s_rts[PART_VEC_SIZE];
    __shared__ uint32_t s_reduction[BLOCK_DIM / LANE_COUNT];

    //full
    if (blockIdx.x < gridDim.x - 1)
    {
        ScanInclusiveFull(
            scan,
            s_reduction,
            s_rts,
            PART_START,
            WARP_PARTITIONS,
            WARP_PART_START);
    }

    //partial
    if (blockIdx.x == gridDim.x - 1)
    {
        ScanInclusivePartial(
            scan,
            s_reduction,
            s_rts,
            PART_START,
            WARP_PARTITIONS,
            WARP_PART_START,
            vectorizedSize);
    }

    uint32_t prevReduction = blockIdx.x ? threadBlockReductions[blockIdx.x] : 0;
    __syncthreads();

    LocalReduce(s_reduction);
    __syncthreads();

    if (threadIdx.x >= LANE_COUNT)
        prevReduction += s_reduction[WARP_INDEX - 1];

    if (blockIdx.x < gridDim.x - 1)
    {
        DownSweepFull(
            scan,
            s_rts,
            prevReduction,
            PART_START,
            WARP_PARTITIONS,
            WARP_PART_START);
    }


    if (blockIdx.x == gridDim.x - 1)
    {
        DownSweepPartial(
            scan,
            s_rts,
            prevReduction,
            PART_START,
            WARP_PARTITIONS,
            WARP_PART_START,
            vectorizedSize);
    }
}