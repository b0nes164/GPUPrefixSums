/******************************************************************************
 * GPUPrefixSums
 * Reduce-Then-Scan
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 12/2/2024
 * https://github.com/b0nes164/GPUPrefixSums
 * 
 ******************************************************************************/
#include "ScanCommon.hlsl"
RWStructuredBuffer<uint> b_threadBlockReduction : register(u2);

//Reduce elements
inline void WaveReduceFull(uint gtid, uint gid)
{
    uint waveReduction = 0;
    [unroll]
    for (uint i = gtid + PartStart(gid), k = 0; k < UINT4_PER_THREAD; i += BLOCK_DIM, ++k)
        waveReduction += WaveActiveSum(dot(b_scanIn[i], uint4(1, 1, 1, 1)));
        
    if (!WaveGetLaneIndex())
        g_reduction[getWaveIndex(gtid)] = waveReduction;
}

inline void WaveReducePartial(uint gtid, uint gid)
{
    uint waveReduction = 0;
    [unroll]
    for (uint i = gtid + PartStart(gid), k = 0; k < UINT4_PER_THREAD; i += BLOCK_DIM, ++k)
    {
        waveReduction += WaveActiveSum(
            i < e_vectorizedSize ? dot(b_scanIn[i], uint4(1, 1, 1, 1)) : 0);
    }
        
    if (!WaveGetLaneIndex())
        g_reduction[getWaveIndex(gtid)] = waveReduction;
}

//reduce warp reductions and write to device memory
inline void LocalReduce(uint gtid, uint gid)
{
    const uint laneLog = countbits(WaveGetLaneCount() - 1);
    const uint spineSize = BLOCK_DIM >> laneLog;
    const uint alignedSize = 1 << (countbits(spineSize - 1) + laneLog - 1) / laneLog * laneLog;
    uint offset = 0;
    for (uint j = laneLog; j <= alignedSize; j <<= laneLog)
    {
        const uint i = (gtid + 1 << offset) - 1;
        const bool pred = i < spineSize;
        const uint t0 = pred ? g_reduction[i] : 0;
        const uint t1 = WaveActiveSum(t0);
        if (pred)
            g_reduction[i] = t1;
        GroupMemoryBarrierWithGroupSync();
        offset += laneLog;
    }
    
    if(!gtid)
        b_threadBlockReduction[gid] = g_reduction[BLOCK_DIM / WaveGetLaneCount() - 1];
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
    
    LocalReduce(gtid.x, flattenedGid);
}

[numthreads(BLOCK_DIM, 1, 1)]
void Scan(uint3 gtid : SV_GroupThreadID)
{
    uint prev = 0;
    const uint alignedSize = (e_threadBlocks + BLOCK_DIM - 1) / BLOCK_DIM * BLOCK_DIM;
    for (uint partStart = 0; partStart < alignedSize; partStart += BLOCK_DIM)
    {
        const uint i = gtid.x + partStart;
        uint t_s = i < e_threadBlocks ? b_threadBlockReduction[i] : 0;
        t_s += WavePrefixSum(t_s);
        
        if (WaveGetLaneIndex() == WaveGetLaneCount() - 1)
            g_reduction[getWaveIndex(gtid.x)] = t_s;
        GroupMemoryBarrierWithGroupSync();
        
        SpineScan(gtid.x);
        GroupMemoryBarrierWithGroupSync();
        
        t_s += prev + (gtid.x >= WaveGetLaneCount() ? g_reduction[getWaveIndex(gtid.x) - 1] : 0);
        if (i < e_threadBlocks)
            b_threadBlockReduction[i] = t_s;
        
        prev += g_reduction[BLOCK_DIM / WaveGetLaneCount() - 1];
        GroupMemoryBarrierWithGroupSync();
    }
}

[numthreads(BLOCK_DIM, 1, 1)]
void PropagateInclusive(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint flattenedGid = flattenGid(gid);
    
    t_scan t_s;
    if (flattenedGid < e_threadBlocks - 1)
        ScanInclusiveFull(gtid.x, flattenedGid, t_s);
    
    if (flattenedGid == e_threadBlocks - 1)
        ScanInclusivePartial(gtid.x, flattenedGid, t_s);
    GroupMemoryBarrierWithGroupSync();
    
    SpineScan(gtid.x);
    GroupMemoryBarrierWithGroupSync();
    
    const uint prevReduction = (flattenedGid ? b_threadBlockReduction[flattenedGid - 1] : 0) +
        (gtid.x >= WaveGetLaneCount() ? g_reduction[getWaveIndex(gtid.x) - 1] : 0);
    
    if (flattenedGid < e_threadBlocks - 1)
        PropagateFull(gtid.x, flattenedGid, prevReduction, t_s);
    
    if (flattenedGid == e_threadBlocks - 1)
        PropagatePartial(gtid.x, flattenedGid, prevReduction, t_s);
}

[numthreads(BLOCK_DIM, 1, 1)]
void PropagateExclusive(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint flattenedGid = flattenGid(gid);
    
    t_scan t_s;
    if (flattenedGid < e_threadBlocks - 1)
        ScanExclusiveFull(gtid.x, flattenedGid, t_s);
    
    if (flattenedGid == e_threadBlocks - 1)
        ScanExclusivePartial(gtid.x, flattenedGid, t_s);
    GroupMemoryBarrierWithGroupSync();
    
    SpineScan(gtid.x);
    GroupMemoryBarrierWithGroupSync();
    
    const uint prevReduction = (flattenedGid ? b_threadBlockReduction[flattenedGid - 1] : 0) +
        (gtid.x >= WaveGetLaneCount() ? g_reduction[getWaveIndex(gtid.x) - 1] : 0);
    
    if (flattenedGid < e_threadBlocks - 1)
        PropagateFull(gtid.x, flattenedGid, prevReduction, t_s);
    
    if (flattenedGid == e_threadBlocks - 1)
        PropagatePartial(gtid.x, flattenedGid, prevReduction, t_s);
}