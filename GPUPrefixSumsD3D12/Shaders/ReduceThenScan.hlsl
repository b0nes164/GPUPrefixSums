/******************************************************************************
 * GPUPrefixSums
 * Reduce-Then-Scan
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 3/5/2024
 * https://github.com/b0nes164/GPUPrefixSums
 * 
 ******************************************************************************/
#include "ScanCommon.hlsl"
RWStructuredBuffer<uint> b_threadBlockReduction : register(u1);
groupshared uint g_scan[BLOCK_DIM];

//Reduce elements
inline void WaveReduceFull(uint gtid, uint gid)
{
    uint waveReduction = 0;
    const uint partEnd = (gid + 1) * UINT4_PART_SIZE;
    for (uint i = gtid + PartStart(gid); i < partEnd; i += BLOCK_DIM)
        waveReduction += WaveActiveSum(dot(b_scan[i], uint4(1, 1, 1, 1)));
        
    if (!WaveGetLaneIndex())
        g_reduction[getWaveIndex(gtid)] = waveReduction;
}

inline void WaveReducePartial(uint gtid, uint gid)
{
    uint waveReduction = 0;
    const uint partEnd = (gid + 1) * UINT4_PART_SIZE;
    for (uint i = gtid + PartStart(gid); i < partEnd; i += BLOCK_DIM)
    {
        waveReduction += WaveActiveSum(
            i < e_vectorizedSize ? dot(b_scan[i], uint4(1, 1, 1, 1)) : 0);
    }
        
    if (!WaveGetLaneIndex())
        g_reduction[getWaveIndex(gtid)] = waveReduction;
}

//reduce warp reductions and write to device memory
inline void LocalReduceWGE16(uint gtid, uint gid)
{
    uint blockReduction;
    if (gtid < BLOCK_DIM / WaveGetLaneCount())
        blockReduction = WaveActiveSum(g_reduction[gtid]);
    
    if (!gtid)
        b_threadBlockReduction[gid] = blockReduction;
}

inline void LocalReduceWLT16(uint gtid, uint gid)
{
    const uint reductionSize = BLOCK_DIM / WaveGetLaneCount();
    if (gtid < reductionSize)
        g_reduction[gtid] = WaveActiveSum(g_reduction[gtid]);
    GroupMemoryBarrierWithGroupSync();
        
    const uint laneLog = countbits(WaveGetLaneCount() - 1);
    uint offset = laneLog;
    uint j = WaveGetLaneCount();
    for (; j < (reductionSize >> 1); j <<= laneLog)
    {
        if (gtid < (reductionSize >> offset))
        {
            g_reduction[((gtid + 1) << offset) - 1] =
                    WaveActiveSum(g_reduction[((gtid + 1) << offset) - 1]);
        }
        GroupMemoryBarrierWithGroupSync();
        offset += laneLog;
    }
        
    if (gtid == reductionSize - 1)
        b_threadBlockReduction[gid] = g_reduction[gtid];
}

inline uint ThreadBlockScanFullWGE16(uint gtid, uint laneMask)
{
    uint reduction = 0;
    const uint partEnd = e_threadBlocks / BLOCK_DIM * BLOCK_DIM;
    for (uint i = gtid; i < partEnd; i += BLOCK_DIM)
    {
        g_scan[gtid] = b_threadBlockReduction[i];
        g_scan[gtid] += WavePrefixSum(g_scan[gtid]);
        GroupMemoryBarrierWithGroupSync();
        
        if (gtid < BLOCK_DIM / WaveGetLaneCount())
        {
            g_scan[(gtid + 1) * WaveGetLaneCount() - 1] +=
                WavePrefixSum(g_scan[(gtid + 1) * WaveGetLaneCount() - 1]);
        }
        GroupMemoryBarrierWithGroupSync();
        
        b_threadBlockReduction[i] = g_scan[gtid] +
            (gtid >= WaveGetLaneCount() && WaveGetLaneIndex() != laneMask ?
            WaveReadLaneAt(g_scan[gtid - 1], 0) : 0) + reduction;
        
        reduction += WaveReadLaneAt(g_scan[BLOCK_DIM - 1], 0);
        GroupMemoryBarrierWithGroupSync();
    }
    
    return reduction;
}

inline void ThreadBlockScanPartialWGE16(uint gtid, uint laneMask, uint reduction)
{
    const uint i = gtid + e_threadBlocks / BLOCK_DIM * BLOCK_DIM;
    if (i < e_threadBlocks)
        g_scan[gtid] = b_threadBlockReduction[i];
    g_scan[gtid] += WavePrefixSum(g_scan[gtid]);
    GroupMemoryBarrierWithGroupSync();
    
    if (gtid < BLOCK_DIM / WaveGetLaneCount())
    {
        g_scan[(gtid + 1) * WaveGetLaneCount() - 1] +=
                WavePrefixSum(g_scan[(gtid + 1) * WaveGetLaneCount() - 1]);
    }
    GroupMemoryBarrierWithGroupSync();
    
    if(i < e_threadBlocks)
    {
        b_threadBlockReduction[i] = g_scan[gtid] +
            (gtid >= WaveGetLaneCount() && WaveGetLaneIndex() != laneMask ?
            WaveReadLaneAt(g_scan[gtid - 1], 0) : 0) + reduction;
    }
}

inline uint ThreadBlockScanFullWLT16(uint gtid, uint laneMask, uint laneLog)
{
    uint reduction = 0;
    const uint partitions = e_threadBlocks / BLOCK_DIM;
    for (uint k = 0; k < partitions; ++k)
    {
        g_scan[gtid] = b_threadBlockReduction[gtid + k * BLOCK_DIM];
        g_scan[gtid] += WavePrefixSum(g_scan[gtid]);
        
        if (gtid < WaveGetLaneCount())
            b_threadBlockReduction[gtid + k * BLOCK_DIM] = g_scan[gtid] + reduction;
        GroupMemoryBarrierWithGroupSync();
        
        uint offset = laneLog;
        uint j = WaveGetLaneCount();
        for (; j < (BLOCK_DIM >> 1); j <<= laneLog)
        {
            if (gtid < (BLOCK_DIM >> offset))
            {
                g_scan[((gtid + 1) << offset) - 1] +=
                    WavePrefixSum(g_scan[((gtid + 1) << offset) - 1]);
            }
            GroupMemoryBarrierWithGroupSync();
            
            if ((gtid & ((j << laneLog) - 1)) >= j)
            {
                if (gtid < (j << laneLog))
                {
                    b_threadBlockReduction[gtid + k * BLOCK_DIM] = g_scan[gtid] +
                        ((gtid + 1) & ((1 << offset) - 1) ?
                        WaveReadLaneAt(g_scan[((gtid >> offset) << offset) - 1], 0) : 0) +
                        reduction;
                }
                else
                {
                    if ((gtid + 1) & ((1 << offset) - 1))
                    {
                        g_scan[gtid] +=
                            WaveReadLaneAt(g_scan[((gtid >> offset) << offset) - 1], 0);
                    }
                }
            }
            offset += laneLog;
        }
        GroupMemoryBarrierWithGroupSync();
        
        //If RADIX is not a multiple of lanecount
        const uint index = gtid + j;
        if (index < BLOCK_DIM)
        {
            b_threadBlockReduction[index + k * BLOCK_DIM] = g_scan[index] +
                ((index + 1) & ((1 << offset) - 1) ?
                WaveReadLaneAt(g_scan[((index >> offset) << offset) - 1], 0) : 0) +
                reduction;
        }
        
        reduction += WaveReadLaneAt(g_scan[BLOCK_DIM - 1] +
            g_scan[((BLOCK_DIM - 1 >> offset) << offset) - 1], 0);
        GroupMemoryBarrierWithGroupSync();
    }
    
    return reduction;
}

inline void ThreadBlockScanPartialWLT16(uint gtid, uint reduction, uint laneMask, uint laneLog)
{
    const uint deviceOffset = e_threadBlocks / BLOCK_DIM * BLOCK_DIM;
    const uint finalPartSize = e_threadBlocks - deviceOffset;
    
    if(gtid < finalPartSize)
        g_scan[gtid] = b_threadBlockReduction[gtid + deviceOffset];
    g_scan[gtid] += WavePrefixSum(g_scan[gtid]);
    
    if(gtid < WaveGetLaneCount() && gtid < finalPartSize)
        b_threadBlockReduction[gtid + deviceOffset] = g_scan[gtid] + reduction;
    GroupMemoryBarrierWithGroupSync();
    
    uint offset = laneLog;
    for (uint j = WaveGetLaneCount(); j < finalPartSize; j <<= laneLog)
    {
        if (gtid < (finalPartSize >> offset))
        {
            g_scan[((gtid + 1) << offset) - 1] +=
                    WavePrefixSum(g_scan[((gtid + 1) << offset) - 1]);
        }
        GroupMemoryBarrierWithGroupSync();
            
        if ((gtid & ((j << laneLog) - 1)) >= j && gtid < finalPartSize)
        {
            if (gtid < (j << laneLog))
            {
                b_threadBlockReduction[gtid + deviceOffset] = g_scan[gtid] +
                    ((gtid + 1) & ((1 << offset) - 1) ?
                    WaveReadLaneAt(g_scan[((gtid >> offset) << offset) - 1], 0) : 0) +
                    reduction;
            }
            else
            {
                if ((gtid + 1) & ((1 << offset) - 1))
                {
                    g_scan[gtid] +=
                        WaveReadLaneAt(g_scan[((gtid >> offset) << offset) - 1], 0);
                }
            }
        }
        offset += laneLog;
    }
}

[numthreads(BLOCK_DIM, 1, 1)]
void Reduce(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint flattenedGid = flattenGid(gid);
    
    if (flattenedGid < e_threadBlocks - 1)
        WaveReduceFull(gtid.x, flattenedGid);
    
    if (flattenedGid == e_threadBlocks - 1)
        WaveReducePartial(gtid.x, flattenedGid);
    GroupMemoryBarrierWithGroupSync();
    
    if (WaveGetLaneCount() >= 16)
        LocalReduceWGE16(gtid.x, flattenedGid);
    
    if (WaveGetLaneCount() < 16)
        LocalReduceWLT16(gtid.x, flattenedGid);
}

[numthreads(BLOCK_DIM, 1, 1)]
void Scan(uint3 gtid : SV_GroupThreadID)
{
    const uint laneMask = WaveGetLaneCount() - 1;
    const uint laneLog = countbits(laneMask);
    
    if (WaveGetLaneCount() >= 16)
    {
        uint reduction = ThreadBlockScanFullWGE16(gtid.x, laneMask);
        ThreadBlockScanPartialWGE16(gtid.x, laneMask, reduction);
    }
    
    if (WaveGetLaneCount() < 16)
    {
        uint reduction = ThreadBlockScanFullWLT16(gtid.x, laneMask, laneLog);
        ThreadBlockScanPartialWLT16(gtid.x, reduction, laneMask, laneLog);
    }
}

[numthreads(BLOCK_DIM, 1, 1)]
void DownsweepInclusive(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint flattenedGid = flattenGid(gid);
    
    if (flattenedGid < e_threadBlocks - 1)
        ScanInclusiveFull(gtid.x, flattenedGid);
    
    if (flattenedGid == e_threadBlocks - 1)
        ScanInclusivePartial(gtid.x, flattenedGid);
    
    uint prevReduction = flattenedGid ? b_threadBlockReduction[flattenedGid - 1] : 0;
    GroupMemoryBarrierWithGroupSync();
    
    if (WaveGetLaneCount() >= 16)
        LocalScanInclusiveWGE16(gtid.x, flattenedGid);
    
    if (WaveGetLaneCount() < 16)
        LocalScanInclusiveWLT16(gtid.x, flattenedGid);
    GroupMemoryBarrierWithGroupSync();
    
    prevReduction += gtid.x >= WaveGetLaneCount() ? g_reduction[getWaveIndex(gtid.x) - 1] : 0;
    
    
    if (flattenedGid < e_threadBlocks - 1)
        DownSweepFull(gtid.x, flattenedGid, prevReduction);
    
    if (flattenedGid == e_threadBlocks - 1)
        DownSweepPartial(gtid.x, flattenedGid, prevReduction);
}

[numthreads(BLOCK_DIM, 1, 1)]
void DownsweepExclusive(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint flattenedGid = flattenGid(gid);
    
    if (flattenedGid < e_threadBlocks - 1)
        ScanExclusiveFull(gtid.x, flattenedGid);
    
    if (flattenedGid == e_threadBlocks - 1)
        ScanExclusivePartial(gtid.x, flattenedGid);
    
    uint prevReduction = flattenedGid ? b_threadBlockReduction[flattenedGid - 1] : 0;
    GroupMemoryBarrierWithGroupSync();
    
    if (WaveGetLaneCount() >= 16)
        LocalScanInclusiveWGE16(gtid.x, flattenedGid);
    
    if (WaveGetLaneCount() < 16)
        LocalScanInclusiveWLT16(gtid.x, flattenedGid);
    GroupMemoryBarrierWithGroupSync();
    
    prevReduction += gtid.x >= WaveGetLaneCount() ? g_reduction[getWaveIndex(gtid.x) - 1] : 0;
    
    if (flattenedGid < e_threadBlocks - 1)
        DownSweepFull(gtid.x, flattenedGid, prevReduction);
    
    if (flattenedGid == e_threadBlocks - 1)
        DownSweepPartial(gtid.x, flattenedGid, prevReduction);
}