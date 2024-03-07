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

#define VECTOR_MASK		3
#define VECTOR_LOG		2

#define WARP_PARTITIONS 3
#define WARP_PART_SIZE  96
#define WARP_PART_START (WARP_INDEX * WARP_PART_SIZE)
#define PART_START      (_partIndex * PART_VEC_SIZE)

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
    __syncthreads();

    return _broadcast;
}

__device__ __forceinline__ void ScanExclusiveFull(
    uint32_t* _scan,
    uint32_t* _reduction,
    uint4* _csdl,
    const uint32_t& _partIndex)
{
    uint32_t warpReduction = 0;

    #pragma unroll
    for (uint32_t i = getLaneId() + WARP_PART_START, k = 0;
        k < WARP_PARTITIONS;
        i += LANE_COUNT, ++k)
    {
        uint4 t = reinterpret_cast<uint4*>(_scan)[i + PART_START];

        uint32_t t2 = t.x;
        t.x += t.y;
        t.y = t2;

        t2 = t.x;
        t.x += t.z;
        t.z = t2;

        t2 = t.x;
        t.x += t.w;
        t.w = t2;

        t2 = InclusiveWarpScanCircularShift(t.x);
        _csdl[i] = SetXAddYZW((getLaneId() ? t2 : 0) + (k ? warpReduction : 0), t);
        warpReduction += __shfl_sync(0xffffffff, t2, 0);
    }

    if (getLaneId() == 0)
        _reduction[WARP_INDEX] = warpReduction;
}

__device__ __forceinline__ void ScanExclusivePartial(
    uint32_t* _scan,
    uint32_t* _reduction,
    uint4* _csdl,
    const uint32_t& _partIndex,
    const uint32_t& alignedSize)
{
    uint32_t warpReduction = 0;
    const uint32_t finalPartSize = alignedSize - PART_START;
    #pragma unroll
    for (uint32_t i = getLaneId() + WARP_PART_START, k = 0;
        k < WARP_PARTITIONS;
        i += LANE_COUNT, ++k)
    {
        uint4 t = i < finalPartSize ? reinterpret_cast<uint4*>(_scan)[i + PART_START] :
            make_uint4(0, 0, 0, 0);

        uint32_t t2 = t.x;
        t.x += t.y;
        t.y = t2;

        t2 = t.x;
        t.x += t.z;
        t.z = t2;

        t2 = t.x;
        t.x += t.w;
        t.w = t2;

        t2 = InclusiveWarpScanCircularShift(t.x);
        _csdl[i] = SetXAddYZW((getLaneId() ? t2 : 0) + (k ? warpReduction : 0), t);
        warpReduction += __shfl_sync(0xffffffff, t2, 0);
    }

    if (getLaneId() == 0)
        _reduction[WARP_INDEX] = warpReduction;
}

__device__ __forceinline__ void ScanInclusiveFull(
    uint32_t* _scan,
    uint32_t* _reduction,
    uint4* _csdl,
    const uint32_t& _partIndex)
{
    uint32_t warpReduction = 0;

    #pragma unroll
    for (uint32_t i = getLaneId() + WARP_PART_START, k = 0;
        k < WARP_PARTITIONS;
        i += LANE_COUNT, ++k)
    {
        uint4 t = reinterpret_cast<uint4*>(_scan)[i + PART_START];
        t.y += t.x;
        t.z += t.y;
        t.w += t.z;

        const uint32_t t2 = InclusiveWarpScanCircularShift(t.w);
        _csdl[i] = AddUintToUint4((getLaneId() ? t2 : 0) + (k ? warpReduction : 0), t);
        warpReduction += __shfl_sync(0xffffffff, t2, 0);
    }

    if (getLaneId() == 0)
        _reduction[WARP_INDEX] = warpReduction;
}

__device__ __forceinline__ void ScanInclusivePartial(
    uint32_t* _scan,
    uint32_t* _reduction,
    uint4* _csdl,
    const uint32_t& _partIndex,
    const uint32_t& alignedSize)
{
    uint32_t warpReduction = 0;
    const uint32_t finalPartSize = alignedSize - PART_START;
    #pragma unroll
    for (uint32_t i = getLaneId() + WARP_PART_START, k = 0;
        k < WARP_PARTITIONS;
        i += LANE_COUNT, ++k)
    {
        uint4 t = i < finalPartSize ? reinterpret_cast<uint4*>(_scan)[i + PART_START] :
            make_uint4(0, 0, 0, 0);
        t.y += t.x;
        t.z += t.y;
        t.w += t.z;

        const uint32_t t2 = InclusiveWarpScanCircularShift(t.w);
        _csdl[i] = AddUintToUint4((getLaneId() ? t2 : 0) + (k ? warpReduction : 0), t);
        warpReduction += __shfl_sync(0xffffffff, t2, 0);
    }

    if (getLaneId() == 0)
        _reduction[WARP_INDEX] = warpReduction;
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

__device__ __forceinline__ void DownSweepFull(
    uint32_t* _scan,
    uint4* _csdl,
    const uint32_t& _prevReduction,
    const uint32_t& _partIndex)
{
    #pragma unroll
    for (uint32_t i = getLaneId() + WARP_PART_START, k = 0;
        k < WARP_PARTITIONS;
        i += LANE_COUNT, ++k)
    {
        reinterpret_cast<uint4*>(_scan)[i + PART_START] =
            AddUintToUint4(_prevReduction, _csdl[i]);
    }
}

__device__ __forceinline__ void DownSweepPartial(
    uint32_t* _scan,
    uint4* _csdl,
    const uint32_t& _prevReduction,
    const uint32_t& _partIndex,
    const uint32_t& _alignedSize)
{
    const uint32_t finalPartSize = _alignedSize - PART_START;
    for (uint32_t i = getLaneId() + WARP_PART_START, k = 0;
        k < WARP_PARTITIONS && i < finalPartSize;
        i += LANE_COUNT, ++k)
    {
        reinterpret_cast<uint4*>(_scan)[i + PART_START] = 
            AddUintToUint4(_prevReduction, _csdl[i]);
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

    const uint32_t partitionIndex = AcquirePartitionIndex(s_broadcast, index);

    if (partitionIndex < gridDim.x - 1)
        ScanExclusiveFull(scan, s_reduction, s_csdl, partitionIndex);

    if (partitionIndex == gridDim.x - 1)
        ScanExclusivePartial(scan, s_reduction, s_csdl, partitionIndex, alignedSize);
    __syncthreads();

    LocalReduceDeviceBroadcast(s_reduction, threadBlockReduction, partitionIndex);
        
    uint32_t prevReduction = 0;
    if (partitionIndex && threadIdx.x < LANE_COUNT)
        Lookback(s_broadcast, threadBlockReduction, partitionIndex, prevReduction);
    __syncthreads();
    
    if (threadIdx.x >= LANE_COUNT)
        prevReduction += s_broadcast + s_reduction[WARP_INDEX - 1];

    if (partitionIndex < gridDim.x - 1)
        DownSweepFull(scan, s_csdl, prevReduction, partitionIndex);

    if (partitionIndex == gridDim.x - 1)
        DownSweepPartial(scan, s_csdl, prevReduction, partitionIndex, alignedSize);
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

    const uint32_t partitionIndex = AcquirePartitionIndex(s_broadcast, index);

    if (partitionIndex < gridDim.x - 1)
        ScanInclusiveFull(scan, s_reduction, s_csdl, partitionIndex);

    if (partitionIndex == gridDim.x - 1)
        ScanInclusivePartial(scan, s_reduction, s_csdl, partitionIndex, alignedSize);
    __syncthreads();

    LocalReduceDeviceBroadcast(s_reduction, threadBlockReduction, partitionIndex);

    uint32_t prevReduction = 0;
    if (partitionIndex && threadIdx.x < LANE_COUNT)
        Lookback(s_broadcast, threadBlockReduction, partitionIndex, prevReduction);
    __syncthreads();

    if (threadIdx.x >= LANE_COUNT)
        prevReduction += s_broadcast + s_reduction[WARP_INDEX - 1];

    if (partitionIndex < gridDim.x - 1)
        DownSweepFull(scan, s_csdl, prevReduction, partitionIndex);

    if (partitionIndex == gridDim.x - 1)
        DownSweepPartial(scan, s_csdl, prevReduction, partitionIndex, alignedSize);
}