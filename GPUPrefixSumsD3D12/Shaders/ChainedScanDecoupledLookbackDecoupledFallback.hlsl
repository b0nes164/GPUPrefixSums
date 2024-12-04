/******************************************************************************
 * GPUPrefixSums
 * Chained Scan Decoupled Lookback Decoupled Fallback
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 12/2/2024
 * https://github.com/b0nes164/GPUPrefixSums
 * 
 ******************************************************************************/
#include "ScanCommon.hlsl"

//For the lookback
#define FLAG_NOT_READY  0           //Flag indicating this partition tile's local reduction is not ready
#define FLAG_REDUCTION  1           //Flag indicating this partition tile's local reduction is ready
#define FLAG_INCLUSIVE  2           //Flag indicating this partition tile has summed all preceding tiles and added to its sum.
#define FLAG_MASK       3           //Mask used to retrieve the flag

//For the fallback
#define MAX_SPIN_COUNT  4           //Max a threadblock is allowed to spin before it performs fallback
#define LOCKED          true
#define UNLOCKED        false

globallycoherent RWStructuredBuffer<uint> b_scanBump : register(u2);
globallycoherent RWStructuredBuffer<uint> b_threadBlockReduction : register(u3);

groupshared uint g_broadcast;
groupshared bool g_lock;
groupshared uint g_fallBackReduction[BLOCK_DIM / MIN_WAVE_SIZE];

inline void AcquirePartitionIndexSetLock(uint gtid)
{
    if (!gtid)
    {
        InterlockedAdd(b_scanBump[0], 1, g_broadcast);
        g_lock = LOCKED;
    }
}

inline void DeviceBroadcast(uint gtid, uint partIndex)
{
    if (!gtid)
    {
        uint t;
        InterlockedExchange(b_threadBlockReduction[partIndex],
            (partIndex ? FLAG_REDUCTION : FLAG_INCLUSIVE) |
            g_reduction[BLOCK_DIM / WaveGetLaneCount() - 1] << 2, t);
    }
}

//Bounds checking is unnecessary because the final partition can never deadlock
inline void WaveReduceFull(uint gtid, uint fallbackIndex)
{
    uint waveReduction = 0;
    [unroll]
    for (uint i = gtid + PartStart(fallbackIndex), k = 0; k < UINT4_PER_THREAD; i += BLOCK_DIM, ++k)
        waveReduction += WaveActiveSum(dot(b_scanIn[i], uint4(1, 1, 1, 1)));
        
    if (!WaveGetLaneIndex())
        g_fallBackReduction[getWaveIndex(gtid)] = waveReduction;
}

inline void LocalReduce(uint gtid)
{
    const uint laneLog = countbits(WaveGetLaneCount() - 1);
    const uint spineSize = BLOCK_DIM >> laneLog;
    const uint alignedSize = 1 << (countbits(spineSize - 1) + laneLog - 1) / laneLog * laneLog;
    uint offset = 0;
    for (uint j = laneLog; j <= alignedSize; j <<= laneLog)
    {
        const uint i = (gtid + 1 << offset) - 1;
        const bool pred = i < spineSize;
        const uint t0 = pred ? g_fallBackReduction[i] : 0;
        const uint t1 = WaveActiveSum(t0);
        if (pred)
            g_fallBackReduction[i] = t1;
        GroupMemoryBarrierWithGroupSync();
        offset += laneLog;
    }
}

inline void LookbackWithFallback(uint gtid, uint partIndex)
{
    uint prevReduction = 0;
    uint lookbackIndex = partIndex - 1;
    while(g_lock == LOCKED)
    {
        GroupMemoryBarrierWithGroupSync();
        
        if (!gtid)
        {
            uint spinCount = 0;
            while(spinCount < MAX_SPIN_COUNT)
            {
                const uint flagPayload = b_threadBlockReduction[lookbackIndex];
                if ((flagPayload & FLAG_MASK) > FLAG_NOT_READY)
                {
                    spinCount = 0;
                    prevReduction += flagPayload >> 2;
                    if ((flagPayload & FLAG_MASK) ==  FLAG_INCLUSIVE)
                    {
                        uint t;
                        InterlockedExchange(b_threadBlockReduction[partIndex], FLAG_INCLUSIVE |
                            prevReduction + g_reduction[BLOCK_DIM / WaveGetLaneCount() - 1] << 2, t);
                        g_broadcast = prevReduction;
                        g_lock = UNLOCKED;
                        break;
                    }
                    else
                    {
                        lookbackIndex--;
                    }
                }
                else
                {
                    spinCount++;
                }
            }
            
            //If we did not complete the lookback within the alotted spins,
            //broadcast the lookback id in shared memory to prepare for the fallback
            if(spinCount == MAX_SPIN_COUNT)
            {
                g_broadcast = lookbackIndex;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        //Fallback if still locked
        if(g_lock == LOCKED)
        {
            const uint fallbackIndex = g_broadcast;
            WaveReduceFull(gtid, fallbackIndex);
            GroupMemoryBarrierWithGroupSync();
            
            LocalReduce(gtid);
            
            if(!gtid)
            {
                const uint fallbackReduction = g_fallBackReduction[BLOCK_DIM / WaveGetLaneCount() - 1];
                uint fallbackPayload;
                InterlockedMax(b_threadBlockReduction[fallbackIndex],
                    (fallbackIndex ? FLAG_REDUCTION : FLAG_INCLUSIVE) | fallbackReduction << 2, fallbackPayload);
                prevReduction += fallbackPayload ? fallbackPayload >> 2 : fallbackReduction;
                if (!fallbackIndex || (fallbackPayload & FLAG_MASK) == FLAG_INCLUSIVE)
                {
                    uint t;
                    InterlockedExchange(b_threadBlockReduction[partIndex], FLAG_INCLUSIVE |
                            prevReduction + g_reduction[BLOCK_DIM / WaveGetLaneCount() - 1] << 2, t);
                    g_broadcast = prevReduction;
                    g_lock = UNLOCKED;
                }
                else
                {
                    lookbackIndex--;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
}

[numthreads(256, 1, 1)]
void InitCSDLDF(uint3 id : SV_DispatchThreadID)
{
    const uint increment = 256 * 256;
    
    for (uint i = id.x; i < e_threadBlocks; i += increment)
        b_threadBlockReduction[i] = 0;
    
    if (!id.x)
        b_scanBump[id.x] = 0;
}

[numthreads(BLOCK_DIM, 1, 1)]
void ChainedScanDecoupledLookbackDecoupledFallbackInclusive(uint gtid : SV_GroupThreadID)
{
    AcquirePartitionIndexSetLock(gtid.x);
    GroupMemoryBarrierWithGroupSync();
    const uint partitionIndex = g_broadcast;

    t_scan t_s;
    if (partitionIndex < e_threadBlocks - 1)
        ScanInclusiveFull(gtid.x, partitionIndex, t_s);
    
    if (partitionIndex == e_threadBlocks - 1)
        ScanInclusivePartial(gtid.x, partitionIndex, t_s);
    GroupMemoryBarrierWithGroupSync();
    
    SpineScan(gtid.x);
    GroupMemoryBarrierWithGroupSync();
    
    DeviceBroadcast(gtid.x, partitionIndex);
    
    if(partitionIndex)
        LookbackWithFallback(gtid.x, partitionIndex);
    
    const uint prevReduction = g_broadcast +
        (gtid.x >= WaveGetLaneCount() ? g_reduction[getWaveIndex(gtid.x) - 1] : 0);
    
    if (partitionIndex < e_threadBlocks - 1)
        PropagateFull(gtid.x, partitionIndex, prevReduction, t_s);
    
    if (partitionIndex == e_threadBlocks - 1)
        PropagatePartial(gtid.x, partitionIndex, prevReduction, t_s);
}

[numthreads(BLOCK_DIM, 1, 1)]
void ChainedScanDecoupledLookbackDecoupledFallbackExclusive(uint gtid : SV_GroupThreadID)
{
    AcquirePartitionIndexSetLock(gtid.x);
    GroupMemoryBarrierWithGroupSync();
    const uint partitionIndex = g_broadcast;

    t_scan t_s;
    if (partitionIndex < e_threadBlocks - 1)
        ScanExclusiveFull(gtid.x, partitionIndex, t_s);
    
    if (partitionIndex == e_threadBlocks - 1)
        ScanExclusivePartial(gtid.x, partitionIndex, t_s);
    GroupMemoryBarrierWithGroupSync();
    
    SpineScan(gtid.x);
    GroupMemoryBarrierWithGroupSync();
    
    DeviceBroadcast(gtid.x, partitionIndex);
    
    if (partitionIndex)
        LookbackWithFallback(gtid.x, partitionIndex);
    
    const uint prevReduction = g_broadcast +
        (gtid.x >= WaveGetLaneCount() ? g_reduction[getWaveIndex(gtid.x) - 1] : 0);
    
    if (partitionIndex < e_threadBlocks - 1)
        PropagateFull(gtid.x, partitionIndex, prevReduction, t_s);
    
    if (partitionIndex == e_threadBlocks - 1)
        PropagatePartial(gtid.x, partitionIndex, prevReduction, t_s);
}