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
#include "ScanCommon.hlsl"

#define FLAG_NOT_READY  0           //Flag indicating this partition tile's local reduction is not ready
#define FLAG_REDUCTION  1           //Flag indicating this partition tile's local reduction is ready
#define FLAG_INCLUSIVE  2           //Flag indicating this partition tile has summed all preceding tiles and added to its sum.
#define FLAG_MASK       3           //Mask used to retrieve the flag

globallycoherent RWStructuredBuffer<uint> b_index                   : register(u1);
globallycoherent RWStructuredBuffer<uint> b_threadBlockReduction    : register(u2);

groupshared uint g_broadcast;

inline void AcquirePartitionIndex(uint gtid)
{
    if(!gtid)
        InterlockedAdd(b_index[0], 1, g_broadcast);
}

//use the exact thread that performed the scan on the last element
//to elide an extra barrier
inline void DeviceBroadcast(uint gtid, uint partIndex)
{
    if (gtid == BLOCK_DIM / WaveGetLaneCount() - 1)
    {
        InterlockedAdd(b_threadBlockReduction[partIndex],
            (partIndex ? FLAG_REDUCTION : FLAG_INCLUSIVE) | g_reduction[gtid] << 2);
    }
}

inline void Lookback(uint partIndex)
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
                    g_broadcast = prevReduction;
                    InterlockedAdd(b_threadBlockReduction[partIndex], 1 | (prevReduction << 2));
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
        b_index[id.x] = 0;
}

[numthreads(BLOCK_DIM, 1, 1)]
void ChainedScanDecoupledLookbackExclusive(uint3 gtid : SV_GroupThreadID)
{    
    AcquirePartitionIndex(gtid.x);
    GroupMemoryBarrierWithGroupSync();
    const uint partitionIndex = g_broadcast;

    if (partitionIndex < e_threadBlocks - 1)
        ScanExclusiveFull(gtid.x, partitionIndex);
    
    if(partitionIndex == e_threadBlocks - 1)
        ScanExclusivePartial(gtid.x, partitionIndex);
    GroupMemoryBarrierWithGroupSync();
    
    if (WaveGetLaneCount() >= 16)
        LocalScanInclusiveWGE16(gtid.x, partitionIndex);
    
    if (WaveGetLaneCount() < 16)
        LocalScanInclusiveWLT16(gtid.x, partitionIndex);
    
    DeviceBroadcast(gtid.x, partitionIndex);
    
    if (partitionIndex && gtid.x < WaveGetLaneCount())
        Lookback(partitionIndex);
    GroupMemoryBarrierWithGroupSync();
    
    const uint prevReduction = g_broadcast + 
        (gtid.x >= WaveGetLaneCount() ? g_reduction[getWaveIndex(gtid.x) - 1] : 0);
    
    if (partitionIndex < e_threadBlocks - 1)
        DownSweepFull(gtid.x, partitionIndex, prevReduction);
    
    if (partitionIndex == e_threadBlocks - 1)
        DownSweepPartial(gtid.x, partitionIndex, prevReduction);
}

[numthreads(BLOCK_DIM, 1, 1)]
void ChainedScanDecoupledLookbackInclusive(uint3 gtid : SV_GroupThreadID)
{
    AcquirePartitionIndex(gtid.x);
    GroupMemoryBarrierWithGroupSync();
    const uint partitionIndex = g_broadcast;

    if (partitionIndex < e_threadBlocks - 1)
        ScanInclusiveFull(gtid.x, partitionIndex);
    
    if (partitionIndex == e_threadBlocks - 1)
        ScanInclusivePartial(gtid.x, partitionIndex);
    GroupMemoryBarrierWithGroupSync();
    
    if (WaveGetLaneCount() >= 16)
        LocalScanInclusiveWGE16(gtid.x, partitionIndex);
    
    if (WaveGetLaneCount() < 16)
        LocalScanInclusiveWLT16(gtid.x, partitionIndex);
    
    DeviceBroadcast(gtid.x, partitionIndex);
    
    if (partitionIndex && gtid.x < WaveGetLaneCount())
        Lookback(partitionIndex);
    GroupMemoryBarrierWithGroupSync();
    
    const uint prevReduction = g_broadcast +
        (gtid.x >= WaveGetLaneCount() ? g_reduction[getWaveIndex(gtid.x) - 1] : 0);
    
    if (partitionIndex < e_threadBlocks - 1)
        DownSweepFull(gtid.x, partitionIndex, prevReduction);
    
    if (partitionIndex == e_threadBlocks - 1)
        DownSweepPartial(gtid.x, partitionIndex, prevReduction);
}