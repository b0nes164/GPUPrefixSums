/******************************************************************************
 * GPUPrefixSums
 * Chained Scan Decoupled Lookback Decoupled Fallback
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 4/12/2024
 * https://github.com/b0nes164/GPUPrefixSums
 * 
 * Based off of idea proposed by Raph Levien:
 *      https://raphlinus.github.io/gpu/2021/11/17/prefix-sum-portable.html
 *
 ******************************************************************************/
#include "ScanCommon.hlsl"

#define FLAG_NOT_READY  0           //Flag indicating this partition tile's local reduction is not ready
#define FLAG_REDUCTION  1           //Flag indicating this partition tile's local reduction is ready
#define FLAG_INCLUSIVE  2           //Flag indicating this partition tile has summed all preceding tiles and added to its sum.
#define FLAG_MASK       3           //Mask used to retrieve the flag

#define MAX_SPIN_COUNT  8           //Max a threadblock is allowed to spin before it performs fallback

globallycoherent RWStructuredBuffer<uint> b_index : register(u1);
globallycoherent RWStructuredBuffer<uint> b_threadBlockReduction : register(u2);

groupshared uint g_broadcast;
groupshared bool g_lock;
groupshared uint g_fallBackReduction[BLOCK_DIM / MIN_WAVE_SIZE];

inline void AcquirePartitionIndex(uint gtid)
{
    if (!gtid)
        InterlockedAdd(b_index[0], 1, g_broadcast);
}

inline void SetLock(uint gtid)
{
    if (!gtid)
        g_lock = true;
}

inline void DeviceBroadcast(uint gtid, uint partIndex)
{
    if (gtid == BLOCK_DIM / WaveGetLaneCount() - 1)
    {
        InterlockedCompareStore(b_threadBlockReduction[partIndex], 0,
            (partIndex ? FLAG_REDUCTION : FLAG_INCLUSIVE) | g_reduction[gtid] << 2);
    }
}

inline void WaveReduceFull(uint gtid, uint gid)
{
    uint waveReduction = 0;
    const uint partEnd = (gid + 1) * UINT4_PART_SIZE;
    for (uint i = gtid + PartStart(gid); i < partEnd; i += BLOCK_DIM)
        waveReduction += WaveActiveSum(dot(b_scan[i], uint4(1, 1, 1, 1)));
        
    if (!WaveGetLaneIndex())
        g_fallBackReduction[getWaveIndex(gtid)] = waveReduction;
}

inline uint LocalReduceWGE16(uint gtid, uint toReduceIndex)
{
    uint blockReduction;
    if (gtid < BLOCK_DIM / WaveGetLaneCount())
        blockReduction = WaveActiveSum(g_fallBackReduction[gtid]);
    return blockReduction;
}

inline uint LocalReduceWLT16(uint gtid, uint toReduceIndex)
{
    const uint reductionSize = BLOCK_DIM / WaveGetLaneCount();
    if (gtid < reductionSize)
        g_fallBackReduction[gtid] = WaveActiveSum(g_fallBackReduction[gtid]);
    GroupMemoryBarrierWithGroupSync();
        
    const uint laneLog = countbits(WaveGetLaneCount() - 1);
    uint offset = laneLog;
    uint j = WaveGetLaneCount();
    for (; j < (reductionSize >> 1); j <<= laneLog)
    {
        if (gtid < (reductionSize >> offset))
        {
            g_fallBackReduction[((gtid + 1) << offset) - 1] =
                    WaveActiveSum(g_fallBackReduction[((gtid + 1) << offset) - 1]);
        }
        GroupMemoryBarrierWithGroupSync();
        offset += laneLog;
    }
    
    uint blockReduction;
    if (!gtid)
        blockReduction = g_fallBackReduction[reductionSize - 1];
    return blockReduction;
}

inline void FallBack(
    uint gtid,
    uint partIndex,
    uint toReduceIndex,
    inout uint prevReduction,
    inout uint spinCount,
    inout uint lookBackIndex)
{
    WaveReduceFull(gtid, toReduceIndex);
    GroupMemoryBarrierWithGroupSync();

    uint blockReduction;
    if (WaveGetLaneCount() >= 16)
        blockReduction = LocalReduceWGE16(gtid, toReduceIndex);
    
    if (WaveGetLaneCount() < 16)
        blockReduction = LocalReduceWLT16(gtid, toReduceIndex);
    
    if (!gtid)
    {
        uint valueOut;
        InterlockedCompareExchange(b_threadBlockReduction[toReduceIndex], 0,
            (toReduceIndex ? FLAG_REDUCTION : FLAG_INCLUSIVE) | blockReduction << 2, valueOut);

        if (!valueOut)
            prevReduction += blockReduction;
        else
            prevReduction += valueOut >> 2;
        
        if (!toReduceIndex || (valueOut & FLAG_MASK) == FLAG_INCLUSIVE)
        {
            g_broadcast = prevReduction;
            g_lock = false;
            InterlockedAdd(b_threadBlockReduction[partIndex], 1 | (prevReduction << 2));
        }
        else
        {
            spinCount = 0;
            lookBackIndex--;
        }
    }
}

inline void LookbackSingleWithFallBack(uint gtid, uint partIndex)
{
    uint spinCount = 0;
    uint prevReduction = 0;
    uint lookBackIndex = partIndex - 1;
    
    while (WaveReadLaneAt(g_lock, 0) == true)
    {
        GroupMemoryBarrierWithGroupSync();
        
        if (!gtid)
        {
            while (spinCount < MAX_SPIN_COUNT)
            {
                const uint flagPayload = b_threadBlockReduction[lookBackIndex];
                
                if ((flagPayload & FLAG_MASK) > FLAG_NOT_READY)
                {
                    prevReduction += flagPayload >> 2;
                    if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE)
                    {
                        g_broadcast = prevReduction;
                        g_lock = false;
                        InterlockedAdd(b_threadBlockReduction[partIndex], 1 | (prevReduction << 2));
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
            
            if (g_lock)
                g_broadcast = lookBackIndex;
        }
        GroupMemoryBarrierWithGroupSync();
        
        if (g_lock)
        {
            FallBack(
                gtid,
                partIndex,
                g_broadcast,
                prevReduction,
                spinCount,
                lookBackIndex);
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

[numthreads(256, 1, 1)]
void InitCSDLDF(uint3 id : SV_DispatchThreadID)
{
    const uint increment = 256 * 256;
    
    for (uint i = id.x; i < e_threadBlocks; i += increment)
        b_threadBlockReduction[i] = 0;
    
    if (!id.x)
        b_index[id.x] = 0;
}

[numthreads(BLOCK_DIM, 1, 1)]
void ChainedScanDecoupledLookbackDecoupledFallbackInclusive(uint gtid : SV_GroupThreadID)
{
    AcquirePartitionIndex(gtid.x);
    SetLock(gtid.x);
    GroupMemoryBarrierWithGroupSync();
    const uint partitionIndex = g_broadcast;

    if (partitionIndex < e_threadBlocks - 1)
        ScanInclusiveFull(gtid.x, partitionIndex);
    
    if (partitionIndex == e_threadBlocks - 1)
        ScanInclusivePartial(gtid.x, partitionIndex);
    GroupMemoryBarrierWithGroupSync();
    
    if (WaveGetLaneCount() >= 16)
        LocalScanInclusiveWGE16(gtid.x);
    
    if (WaveGetLaneCount() < 16)
        LocalScanInclusiveWLT16(gtid.x);
    
    DeviceBroadcast(gtid.x, partitionIndex);

    if (partitionIndex)
        LookbackSingleWithFallBack(gtid.x, partitionIndex);
    else
        GroupMemoryBarrierWithGroupSync();
    
    const uint prevReduction = g_broadcast +
        (gtid.x >= WaveGetLaneCount() ? g_reduction[getWaveIndex(gtid.x) - 1] : 0);
    
    if (partitionIndex < e_threadBlocks - 1)
        DownSweepFull(gtid.x, partitionIndex, prevReduction);
    
    if (partitionIndex == e_threadBlocks - 1)
        DownSweepPartial(gtid.x, partitionIndex, prevReduction);
}

[numthreads(BLOCK_DIM, 1, 1)]
void ChainedScanDecoupledLookbackDecoupledFallbackExclusive(uint gtid : SV_GroupThreadID)
{
    AcquirePartitionIndex(gtid.x);
    SetLock(gtid.x);
    GroupMemoryBarrierWithGroupSync();
    const uint partitionIndex = g_broadcast;

    if (partitionIndex < e_threadBlocks - 1)
        ScanExclusiveFull(gtid.x, partitionIndex);
    
    if (partitionIndex == e_threadBlocks - 1)
        ScanExclusivePartial(gtid.x, partitionIndex);
    GroupMemoryBarrierWithGroupSync();
    
    if (WaveGetLaneCount() >= 16)
        LocalScanInclusiveWGE16(gtid.x);
    
    if (WaveGetLaneCount() < 16)
        LocalScanInclusiveWLT16(gtid.x);
    
    DeviceBroadcast(gtid.x, partitionIndex);

    if (partitionIndex)
        LookbackSingleWithFallBack(gtid.x, partitionIndex);
    else
        GroupMemoryBarrierWithGroupSync();
    
    const uint prevReduction = g_broadcast +
        (gtid.x >= WaveGetLaneCount() ? g_reduction[getWaveIndex(gtid.x) - 1] : 0);
    
    if (partitionIndex < e_threadBlocks - 1)
        DownSweepFull(gtid.x, partitionIndex, prevReduction);
    
    if (partitionIndex == e_threadBlocks - 1)
        DownSweepPartial(gtid.x, partitionIndex, prevReduction);
}