/******************************************************************************
 * GPUPrefixSums
 * Emulated Deadlocking
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 4/15/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#include "EmulatedDeadlocking.cuh"

#define PART_VEC_SIZE	768
#define BLOCK_DIM       256

#define WARP_PARTITIONS 3
#define WARP_PART_SIZE  96
#define WARP_PART_START (WARP_INDEX * WARP_PART_SIZE)
#define PART_START      (partitionIndex * PART_VEC_SIZE)

#define FLAG_NOT_READY  0
#define FLAG_REDUCTION  1
#define FLAG_INCLUSIVE  2
#define FLAG_MASK       3

#define MAX_SPIN_COUNT  8
#define MASK            1

 //Atomically acquire partition index
__device__ __forceinline__ uint32_t AcquirePartitionIndex(
    uint32_t& _broadcast,
    volatile uint32_t* _index)
{
    if (threadIdx.x == 0)
        _broadcast = atomicAdd((uint32_t*)&_index[0], 1);
}

__device__ __forceinline__ uint32_t SetLock(
    uint32_t& _lock)
{
    if (threadIdx.x)
        _lock = true;
}

__device__ __forceinline__ void LocalScanDeviceBroadcastEmulatedDeadlock(
    uint32_t* _reduction,
    volatile uint32_t* _threadBlockReduction,
    const uint32_t& _partIndex)
{
    if (threadIdx.x < BLOCK_DIM / LANE_COUNT)
        _reduction[threadIdx.x] = ActiveInclusiveWarpScan(_reduction[threadIdx.x]);

    if (!(_partIndex & MASK) && _partIndex != gridDim.x - 1)
    {
        while (_threadBlockReduction[_partIndex] == 0)
        {
            __syncthreads();
        }
    }
    else
    {
        if (threadIdx.x == BLOCK_DIM / LANE_COUNT - 1)
        {
            atomicCAS((uint32_t*)&_threadBlockReduction[_partIndex], 0,
                (_partIndex ? FLAG_REDUCTION : FLAG_INCLUSIVE) | _reduction[threadIdx.x] << 2);
        }
    }
}

__device__ __forceinline__ void LocalReduce(
    uint32_t _partIndex,
    volatile uint32_t* _localReduction,
    uint32_t* scan)
{
    uint32_t warpReduction = 0;
    const uint32_t partEnd = (_partIndex + 1) * PART_VEC_SIZE;
    for (uint32_t i = threadIdx.x + (_partIndex * PART_VEC_SIZE); i < partEnd; i += blockDim.x)
    {
        const uint4 t = reinterpret_cast<uint4*>(scan)[i];
        warpReduction += WarpReduceSum(t.x + t.y + t.z + t.w);
    }

    if (!getLaneId())
        _localReduction[WARP_INDEX] = warpReduction;
}

__device__ __forceinline__ void Fallback(
    uint32_t _partIndex,
    uint32_t _toReduceIndex,
    uint32_t& _prevReduction,
    bool& lock,
    uint32_t& broadcast,
    uint32_t& spinCount,
    uint32_t* scan,
    volatile uint32_t* _localReduction,
    volatile uint32_t* _threadBlockReduction)
{
    LocalReduce(_toReduceIndex, _localReduction, scan);
    __syncthreads();

    uint32_t blockReduction;
    if(threadIdx.x < blockDim.x / LANE_COUNT)
        blockReduction = ActiveWarpReduceSum(_localReduction[threadIdx.x]);

    if (!threadIdx.x)
    {
        uint32_t valueOut = atomicCAS((uint32_t*)&_threadBlockReduction[_toReduceIndex], 0,
            (_toReduceIndex ? FLAG_REDUCTION : FLAG_INCLUSIVE) | blockReduction << 2);

        if (!valueOut)
            _prevReduction += blockReduction;
        else
            _prevReduction += valueOut >> 2;

        if (!_toReduceIndex || (valueOut & FLAG_MASK) == FLAG_INCLUSIVE)
        {
            broadcast = _prevReduction;
            lock = false;
            atomicAdd((uint32_t*)&_threadBlockReduction[_partIndex], 1 | (_prevReduction << 2));
        }
        else
        {
            spinCount = 0;
        }
    }
}

__device__ __forceinline__ void Lookback(
    const uint32_t& _partIndex,
    bool& _lock,
    uint32_t& _broadcast,
    uint32_t* scan,
    volatile uint32_t* _localReduction,
    volatile uint32_t* _threadBlockReduction)
{
    uint32_t _prevReduction = 0;
    uint32_t spinCount = 0;
    uint32_t lookBackIndex = _partIndex - 1;

    while (_lock == true)
    {
        __syncthreads();

        if (!threadIdx.x)
        {
            while (spinCount < MAX_SPIN_COUNT)
            {
                const uint32_t flagPayload = _threadBlockReduction[lookBackIndex];

                if ((flagPayload & FLAG_MASK) > FLAG_NOT_READY)
                {
                    _prevReduction += flagPayload >> 2;
                    if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE)
                    {
                        _broadcast = _prevReduction;
                        _lock = false;
                        atomicAdd((uint32_t*)&_threadBlockReduction[_partIndex],
                            1 | (_prevReduction << 2));
                        break;
                    }
                    else
                    {
                        lookBackIndex--;
                    }
                }
                else
                {
                    spinCount++;
                }
            }

            if (_lock)
                _broadcast = lookBackIndex;
        }
        __syncthreads();

        if (_lock)
        {
            Fallback(
                _partIndex,
                _broadcast,
                _prevReduction,
                _lock,
                _broadcast,
                spinCount,
                scan,
                _localReduction,
                _threadBlockReduction);

            if (!threadIdx.x)
                lookBackIndex--;
        }
        __syncthreads();    //Fallback potentially results in an unlock
    }
}

__global__ void EmulatedDeadlocking::EmulatedDeadlockSpinning(
    uint32_t* scan,
    volatile uint32_t* threadBlockReduction,
    volatile uint32_t* index,
    uint32_t vectorizedSize)
{
    __shared__ uint4 s_csdl[PART_VEC_SIZE];
    __shared__ uint32_t s_reduction[BLOCK_DIM / LANE_COUNT];
    __shared__ uint32_t s_localReduction[BLOCK_DIM / LANE_COUNT];
    __shared__ uint32_t s_broadcast;
    __shared__ bool s_lock;

    AcquirePartitionIndex(s_broadcast, index);
    if (!threadIdx.x)
        s_lock = true;
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
            vectorizedSize);
    }
    __syncthreads();

    LocalScanDeviceBroadcastEmulatedDeadlock(s_reduction, threadBlockReduction, partitionIndex);

    if (partitionIndex)
    {
        Lookback(
            partitionIndex,
            s_lock,
            s_broadcast,
            scan,
            s_localReduction,
            threadBlockReduction);
    }
    __syncthreads();

    uint32_t prevReduction = s_broadcast + (threadIdx.x >= LANE_COUNT ? s_reduction[WARP_INDEX - 1] : 0);

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
            vectorizedSize);
    }
}