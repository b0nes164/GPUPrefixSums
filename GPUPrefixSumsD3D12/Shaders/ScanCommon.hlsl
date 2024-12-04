/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 12/2/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#define MAX_DISPATCH_DIM    65535U
#define UINT4_PART_SIZE     768U
#define BLOCK_DIM           256U
#define UINT4_PER_THREAD    3U
#define MIN_WAVE_SIZE       4U

cbuffer cbPrefixSum : register(b0)
{
    uint e_vectorizedSize;
    uint e_threadBlocks;
    uint e_isPartial;
    uint e_fullDispatches;
};

RWStructuredBuffer<uint4> b_scanIn : register(u0);
RWStructuredBuffer<uint4> b_scanOut : register(u1);
groupshared uint g_reduction[BLOCK_DIM / MIN_WAVE_SIZE];
struct t_scan
{
    uint4 t[UINT4_PER_THREAD];
};

inline uint getWaveIndex(uint _gtid)
{
    return _gtid / WaveGetLaneCount();  //CAUTION, 1D WORKGROUP ONLY!
}

inline bool isPartialDispatch()
{
    return e_isPartial;
}

inline uint flattenGid(uint3 gid)
{
    return isPartialDispatch() ?
        gid.x + e_fullDispatches * MAX_DISPATCH_DIM :
        gid.x + gid.y * MAX_DISPATCH_DIM;
}

inline uint PartStart(uint partIndex)
{
    return partIndex * UINT4_PART_SIZE;
}

inline uint WavePartSize()
{
    return UINT4_PER_THREAD * WaveGetLaneCount();
}

inline uint WavePartStart(uint gtid)
{
    return getWaveIndex(gtid) * WavePartSize();
}

inline uint4 SetXAddYZW(uint t, uint4 val)
{
    return uint4(t, val.yzw + t);
}

//read in and scan
inline void ScanExclusiveFull(uint gtid, uint partIndex, inout t_scan t_s)
{
    const uint laneMask = WaveGetLaneCount() - 1;
    const uint circularShift = WaveGetLaneIndex() + laneMask & laneMask;
    uint waveReduction = 0;
    
    [unroll]
    for (uint i = WaveGetLaneIndex() + WavePartStart(gtid) + PartStart(partIndex), k = 0;
        k < UINT4_PER_THREAD;
        i += WaveGetLaneCount(), ++k)
    {
        t_s.t[k] = b_scanIn[i];
        
        uint t0 = t_s.t[k].x;
        t_s.t[k].x += t_s.t[k].y;
        t_s.t[k].y = t0;

        t0 = t_s.t[k].x;
        t_s.t[k].x += t_s.t[k].z;
        t_s.t[k].z = t0;

        t0 = t_s.t[k].x;
        t_s.t[k].x += t_s.t[k].w;
        t_s.t[k].w = t0;
        
        const uint t1 = WaveReadLaneAt(t_s.t[k].x + WavePrefixSum(t_s.t[k].x), circularShift);
        t_s.t[k] = SetXAddYZW((WaveGetLaneIndex() ? t1 : 0) + waveReduction, t_s.t[k]);
        waveReduction += WaveReadLaneAt(t1, 0);
    }
    
    if (!WaveGetLaneIndex())
        g_reduction[getWaveIndex(gtid)] = waveReduction;
}

inline void ScanExclusivePartial(uint gtid, uint partIndex, inout t_scan t_s)
{
    const uint laneMask = WaveGetLaneCount() - 1;
    const uint circularShift = WaveGetLaneIndex() + laneMask & laneMask;
    uint waveReduction = 0;
    
    [unroll]
    for (uint i = WaveGetLaneIndex() + WavePartStart(gtid) + PartStart(partIndex), k = 0;
        k < UINT4_PER_THREAD;
        i += WaveGetLaneCount(), ++k)
    {
        t_s.t[k] = i < e_vectorizedSize ? b_scanIn[i] : 0;
        
        uint t0 = t_s.t[k].x;
        t_s.t[k].x += t_s.t[k].y;
        t_s.t[k].y = t0;

        t0 = t_s.t[k].x;
        t_s.t[k].x += t_s.t[k].z;
        t_s.t[k].z = t0;

        t0 = t_s.t[k].x;
        t_s.t[k].x += t_s.t[k].w;
        t_s.t[k].w = t0;
        
        const uint t1 = WaveReadLaneAt(t_s.t[k].x + WavePrefixSum(t_s.t[k].x), circularShift);
        t_s.t[k] = SetXAddYZW((WaveGetLaneIndex() ? t1 : 0) + waveReduction, t_s.t[k]);
        waveReduction += WaveReadLaneAt(t1, 0);
    }
    
    if (!WaveGetLaneIndex())
        g_reduction[getWaveIndex(gtid)] = waveReduction;
}

inline void ScanInclusiveFull(uint gtid, uint partIndex, inout t_scan t_s)
{
    const uint laneMask = WaveGetLaneCount() - 1;
    const uint circularShift = WaveGetLaneIndex() + laneMask & laneMask;
    uint waveReduction = 0;
    
    [unroll]
    for (uint i = WaveGetLaneIndex() + WavePartStart(gtid) + PartStart(partIndex), k = 0;
        k < UINT4_PER_THREAD;
        i += WaveGetLaneCount(), ++k)
    {
        t_s.t[k] = b_scanIn[i];
        t_s.t[k].y += t_s.t[k].x;
        t_s.t[k].z += t_s.t[k].y;
        t_s.t[k].w += t_s.t[k].z;
        
        const uint t = WaveReadLaneAt(t_s.t[k].w + WavePrefixSum(t_s.t[k].w), circularShift);
        t_s.t[k] += (WaveGetLaneIndex() ? t : 0) + waveReduction;
        waveReduction += WaveReadLaneAt(t, 0);
    }
    
    if (!WaveGetLaneIndex())
        g_reduction[getWaveIndex(gtid)] = waveReduction;
}

inline void ScanInclusivePartial(uint gtid, uint partIndex, inout t_scan t_s)
{
    const uint laneMask = WaveGetLaneCount() - 1;
    const uint circularShift = WaveGetLaneIndex() + laneMask & laneMask;
    uint waveReduction = 0;
    
    [unroll]
    for (uint i = WaveGetLaneIndex() + WavePartStart(gtid) + PartStart(partIndex), k = 0;
        k < UINT4_PER_THREAD;
        i += WaveGetLaneCount(), ++k)
    {
        t_s.t[k] = i < e_vectorizedSize ? b_scanIn[i] : 0;
        t_s.t[k].y += t_s.t[k].x;
        t_s.t[k].z += t_s.t[k].y;
        t_s.t[k].w += t_s.t[k].z;
        
        const uint t = WaveReadLaneAt(t_s.t[k].w + WavePrefixSum(t_s.t[k].w), circularShift);
        t_s.t[k] += (WaveGetLaneIndex() ? t : 0) + waveReduction;
        waveReduction += WaveReadLaneAt(t, 0);
    }
    
    if (!WaveGetLaneIndex())
        g_reduction[getWaveIndex(gtid)] = waveReduction;
}

// Non-divergent wave size agnostic scan across wave reductions
inline void SpineScan(uint gtid)
{
    const uint laneLog = countbits(WaveGetLaneCount() - 1);
    const uint spineSize = BLOCK_DIM >> laneLog;
    const uint alignedSize = 1 << (countbits(spineSize - 1) + laneLog - 1) / laneLog * laneLog;
    uint offset = 0;
    for (uint j = WaveGetLaneCount(); j <= alignedSize; j <<= laneLog)
    {
        const uint t0 = j != WaveGetLaneCount() ? 1 : 0;
        const uint i0 = (gtid + t0 << offset) - t0;
        const bool pred0 = i0 < spineSize;
        const uint t1 = pred0 ? g_reduction[i0] : 0;
        const uint t2 = t1 + WavePrefixSum(t1);
        if (pred0)
            g_reduction[i0] = t2;
        GroupMemoryBarrierWithGroupSync();
        
        if (j != WaveGetLaneCount())
        {
            const uint rshift = j >> laneLog;
            const uint i1 = gtid + rshift;
            if ((i1 & j - 1) >= rshift)
            {
                const bool pred1 = i1 < spineSize;
                const uint t3 = pred1 ? g_reduction[((i1 >> offset) << offset) - 1] : 0;
                if (pred1 && (i1 + 1 & rshift - 1) != 0)
                    g_reduction[i1] += t3;
            }
        }
        offset += laneLog;
    }
}

//Pass in previous reductions, and write out
inline void PropagateFull(uint gtid, uint partIndex, uint prevReduction, t_scan t_s)
{
    [unroll]
    for (uint i = WaveGetLaneIndex() + WavePartStart(gtid) + PartStart(partIndex), k = 0;
        k < UINT4_PER_THREAD;
        i += WaveGetLaneCount(), ++k)
    {
        b_scanOut[i] = t_s.t[k] + prevReduction;
    }
}

inline void PropagatePartial(uint gtid, uint partIndex, uint prevReduction, t_scan t_s)
{
    for (uint i = WaveGetLaneIndex() + WavePartStart(gtid) + PartStart(partIndex), k = 0;
        k < UINT4_PER_THREAD && i < e_vectorizedSize;
        i += WaveGetLaneCount(), ++k)
    {
        b_scanOut[i] = t_s.t[k] + prevReduction;
    }
}