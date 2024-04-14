/******************************************************************************
 * GPUPrefixSums
 * Emulated Forward Progress Deadlocking
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 4/12/2024
 * https://github.com/b0nes164/GPUPrefixSums
 * 
 ******************************************************************************/
#include "ScanCommon.hlsl"

#define FLAG_NOT_READY  0           //Flag indicating this partition tile's local reduction is not ready
#define FLAG_REDUCTION  1           //Flag indicating this partition tile's local reduction is ready
#define FLAG_INCLUSIVE  2           //Flag indicating this partition tile has summed all preceding tiles and added to its sum.
#define FLAG_MASK       3           //Mask used to retrieve the flag

#define MAX_SPIN_COUNT  8           //Max a threadblock is allowed to spin before it performs fallback

#define MASK            1

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

//Instead of using InterlockedAdd or InterlockedOr to perform the atomic store, 
//we use InterlockedCompareStore, this is so that 
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

//identical to the reductions in ReduceThenScan,
//but on different shared memory location
//Attempt to store the reduction, if unsuccessful, discard the result
inline void LocalReduceWGE16(uint gtid, uint toReduceIndex, inout uint prevReduction, inout uint successFlag)
{
    uint blockReduction;
    if (gtid < BLOCK_DIM / WaveGetLaneCount())
        blockReduction = WaveActiveSum(g_fallBackReduction[gtid]);
    
    if (!gtid)
    {
        InterlockedCompareExchange(b_threadBlockReduction[toReduceIndex], 0,
            (toReduceIndex ? FLAG_REDUCTION : FLAG_INCLUSIVE) | blockReduction << 2, successFlag);
        
        if (!successFlag && toReduceIndex)
            prevReduction += blockReduction;
    }
}

inline void LocalReduceWLT16(uint gtid, uint toReduceIndex, inout uint prevReduction, inout uint successFlag)
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
    
    if (!gtid)
    {
        const uint blockReduction = g_fallBackReduction[BLOCK_DIM / WaveGetLaneCount() - 1];
        InterlockedCompareExchange(b_threadBlockReduction[toReduceIndex], 0,
            (toReduceIndex ? FLAG_REDUCTION : FLAG_INCLUSIVE) | blockReduction << 2, successFlag);
        
        if (!successFlag && toReduceIndex)
            prevReduction += blockReduction;
    }
}

//If a deadlock is encountered, deadlockee threadblock 
//reduces the tile itself. The last partition can never be a deadlocker,
//so partial tiles do not need to be taken into account.
inline void FallBackReduction(uint gtid, uint toReduceIndex, inout uint prevReduction, inout uint successFlag)
{
    WaveReduceFull(gtid, toReduceIndex);
    GroupMemoryBarrierWithGroupSync();

    if (WaveGetLaneCount() >= 16)
        LocalReduceWGE16(gtid, toReduceIndex, prevReduction, successFlag);
    
    if (WaveGetLaneCount() < 16)
        LocalReduceWLT16(gtid, toReduceIndex, prevReduction, successFlag);
}

//Perform lookback with a single thread, with fallback if a deadlock is encountered.
//Note all threads participate in the fallback.
inline void LookbackSingleWithFallBack(uint gtid, uint partIndex)
{
    uint spinCount = 0;
    uint prevReduction = 0;
    uint lookBackIndex = partIndex - 1;
    
    while (WaveReadLaneAt(g_lock, 0) == true)
    {
        if (!gtid)
        {
            //Wait for other threadblocks 
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
            
            //If the lookback was not sucessful, and we are now performing a fallback
            //so the lookback thread broadcasts the index of the deadlocker tile
            if (g_lock)
                g_broadcast = lookBackIndex;
        }
        GroupMemoryBarrierWithGroupSync();
        
        //is the lookback successful?
        if (g_lock)
        {
            //This flag indicates whether a threadblock successfully inserted its
            //fallback reduction. As each threadblock is free to perform fallbacks at will,
            //there is a possibility that while this threadblock has been performing a fallback,
            //another threadblock has successfully completed its fallback, inserted its reduction, and unlocked
            //the deadlocker threadblock, which then completed its scan. If that is the case, then the reduction
            //calculated by this threadblock is incorrect, and the result should be discarded.
            uint successFlag = 0xffffffff;
            FallBackReduction(gtid, g_broadcast, prevReduction, successFlag);
            
            if (!gtid)
            {
                if (!successFlag && lookBackIndex)
                    lookBackIndex--;
                spinCount = 0;
            }
        }
    }
}

[numthreads(256, 1, 1)]
void InitEmulatedDeadlock(uint3 id : SV_DispatchThreadID)
{
    const uint increment = 256 * 256;
    
    for (uint i = id.x; i < e_threadBlocks; i += increment)
        b_threadBlockReduction[i] = 0;
    
    if (!id.x)
        b_index[id.x] = 0;
}

[numthreads(1, 1, 1)]
void ClearIndex(uint3 id : SV_DispatchThreadID)
{
    b_index[0] = 0;
}

[numthreads(BLOCK_DIM, 1, 1)]
void EmulatedDeadlockFirstPass(uint gtid : SV_GroupThreadID)
{
    AcquirePartitionIndex(gtid.x);
    SetLock(gtid.x);
    GroupMemoryBarrierWithGroupSync();
    const uint partitionIndex = g_broadcast;

    if (partitionIndex & MASK)
    {
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
}

[numthreads(BLOCK_DIM, 1, 1)]
void EmulatedDeadlockSecondPass(uint gtid : SV_GroupThreadID)
{
    AcquirePartitionIndex(gtid.x);
    SetLock(gtid.x);
    GroupMemoryBarrierWithGroupSync();
    const uint partitionIndex = g_broadcast;

    if (!(partitionIndex & MASK))
    {
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
}

/*
inline void DeviceBroadcastEmulateDeadlock(uint gtid, uint partIndex)
{
    //in a non-emulated deadlock, the final partition can never 
    //be the deadlocking partition
    if (!(partIndex % 97) && partIndex != e_threadBlocks - 1)
    {
        //spin until a fallback unlocks us
        while(b_threadBlockReduction[partIndex] < 4)
        {
            DeviceMemoryBarrierWithGroupSync();
        }
    }
    else
    {
        if (gtid == BLOCK_DIM / WaveGetLaneCount() - 1)
        {
            InterlockedCompareStore(b_threadBlockReduction[partIndex], 0,
                (partIndex ? FLAG_REDUCTION : FLAG_INCLUSIVE) | g_reduction[gtid] << 2);
        }
    }
}
*/