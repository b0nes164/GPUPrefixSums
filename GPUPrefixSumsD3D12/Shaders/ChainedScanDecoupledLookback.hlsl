/******************************************************************************
 * GPUPrefixSums
 * Chained Scan with Decoupled Lookback Implementation
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 12/2/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 * Based off of Research by:
 *          Duane Merrill, Nvidia Corporation
 *          Michael Garland, Nvidia Corporation
 *          https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back
 * 
 ******************************************************************************/
#include "ScanCommon.hlsl"

#define FLAG_NOT_READY  0           //Flag indicating this partition tile's local reduction is not ready
#define FLAG_REDUCTION  1           //Flag indicating this partition tile's local reduction is ready
#define FLAG_INCLUSIVE  2           //Flag indicating this partition tile has summed all preceding tiles and added to its sum.
#define FLAG_MASK       3           //Mask used to retrieve the flag

globallycoherent RWStructuredBuffer<uint> b_scanBump                : register(u2);
globallycoherent RWStructuredBuffer<uint> b_threadBlockReduction    : register(u3);

groupshared uint g_broadcast;

inline void AcquirePartitionIndex(uint gtid)
{
    if (!gtid)
        InterlockedAdd(b_scanBump[0], 1, g_broadcast);
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

//Perform lookback with a single thread
inline void LookbackSingle(uint partIndex)
{
    uint prevReduction = 0;
    uint lookBackIndex = partIndex - 1;
    
    while (true)
    {
        const uint flagPayload = b_threadBlockReduction[lookBackIndex];

        if ((flagPayload & FLAG_MASK) > FLAG_NOT_READY)
        {
            prevReduction += flagPayload >> 2;
            if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE)
            {
                uint t;
                InterlockedExchange(b_threadBlockReduction[partIndex], FLAG_INCLUSIVE |
                    prevReduction + g_reduction[BLOCK_DIM / WaveGetLaneCount() - 1] << 2, t);
                g_broadcast = prevReduction;
                break;
            }
            else
            {
                lookBackIndex--;
            }
        }
    }
}

//Perform lookback with a single warp
inline void LookbackWarp(uint partIndex)
{
    uint prevReduction = 0;
    uint k = partIndex + WaveGetLaneCount() - WaveGetLaneIndex();
    const uint waveParts = (WaveGetLaneCount() + 31) / 32;
    
    while (true)
    {
        const uint flagPayload = k > WaveGetLaneCount() ? 
            b_threadBlockReduction[k - WaveGetLaneCount() - 1] : FLAG_INCLUSIVE;

        if (WaveActiveAllTrue((flagPayload & FLAG_MASK) > FLAG_NOT_READY))
        {
            const uint4 inclusiveBallot = WaveActiveBallot((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE);
            
            //dot(inclusiveBallot, uint4(1,1,1,1)) != 0 does not work
            //consider 0xffffffff + 1 + 0xffffffff + 1
            if (inclusiveBallot.x || inclusiveBallot.y || inclusiveBallot.z || inclusiveBallot.w)
            {
                uint inclusiveIndex = 0;
                for (uint wavePart = 0; wavePart < waveParts; ++wavePart)
                {
                    if (countbits(inclusiveBallot[wavePart]))
                    {
                        inclusiveIndex += firstbitlow(inclusiveBallot[wavePart]);
                        break;
                    }
                    else
                    {
                        inclusiveIndex += 32;
                    }
                }
                                    
                prevReduction += WaveActiveSum(WaveGetLaneIndex() <= inclusiveIndex ? (flagPayload >> 2) : 0);
                                
                if (WaveGetLaneIndex() == 0)
                {
                    uint t;
                    InterlockedExchange(b_threadBlockReduction[partIndex], FLAG_INCLUSIVE |
                        prevReduction + g_reduction[BLOCK_DIM / WaveGetLaneCount() - 1] << 2, t);
                    g_broadcast = prevReduction;
                }
                break;
            }
            else
            {
                prevReduction += WaveActiveSum(flagPayload >> 2);
                k -= WaveGetLaneCount();
            }
        }
    }
}

[numthreads(256, 1, 1)]
void InitChainedScan(uint3 id : SV_DispatchThreadID)
{
    const uint increment = 256 * 256;
    
    for (uint i = id.x; i < e_threadBlocks; i += increment)
        b_threadBlockReduction[i] = 0;
    
    if (!id.x)
        b_scanBump[id.x] = 0;
}

[numthreads(BLOCK_DIM, 1, 1)]
void ChainedScanDecoupledLookbackExclusive(uint3 gtid : SV_GroupThreadID)
{
    AcquirePartitionIndex(gtid.x);
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
    
    if (partitionIndex && !gtid.x)
        LookbackSingle(partitionIndex);
    GroupMemoryBarrierWithGroupSync();
    
    const uint prevReduction = g_broadcast +
        (gtid.x >= WaveGetLaneCount() ? g_reduction[getWaveIndex(gtid.x) - 1] : 0);
    
    if (partitionIndex < e_threadBlocks - 1)
        PropagateFull(gtid.x, partitionIndex, prevReduction, t_s);
    
    if (partitionIndex == e_threadBlocks - 1)
        PropagatePartial(gtid.x, partitionIndex, prevReduction, t_s);
}

[numthreads(BLOCK_DIM, 1, 1)]
void ChainedScanDecoupledLookbackInclusive(uint3 gtid : SV_GroupThreadID)
{
    AcquirePartitionIndex(gtid.x);
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
    
    if (partitionIndex && !gtid.x)
        LookbackSingle(partitionIndex);
    GroupMemoryBarrierWithGroupSync();
    
    const uint prevReduction = g_broadcast +
        (gtid.x >= WaveGetLaneCount() ? g_reduction[getWaveIndex(gtid.x) - 1] : 0);
    
    if (partitionIndex < e_threadBlocks - 1)
        PropagateFull(gtid.x, partitionIndex, prevReduction, t_s);
    
    if (partitionIndex == e_threadBlocks - 1)
        PropagatePartial(gtid.x, partitionIndex, prevReduction, t_s);
}