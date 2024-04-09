/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 3/5/2024
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

RWStructuredBuffer<uint4> b_scan : register(u0);

groupshared uint4 g_shared[UINT4_PART_SIZE];
groupshared uint g_reduction[BLOCK_DIM / MIN_WAVE_SIZE];

inline uint getWaveIndex(uint _gtid)
{
    return _gtid / WaveGetLaneCount();
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

inline uint PartStart(uint _partIndex)
{
    return _partIndex * UINT4_PART_SIZE;
}

inline uint WavePartSize()
{
    return UINT4_PER_THREAD * WaveGetLaneCount();
}

inline uint WavePartStart(uint _gtid)
{
    return getWaveIndex(_gtid) * WavePartSize();
}

inline uint4 SetXAddYZW(uint t, uint4 val)
{
    return uint4(t, val.yzw + t);
}

//read in and scan
inline void ScanExclusiveFull(uint gtid, uint partIndex)
{
    const uint laneMask = WaveGetLaneCount() - 1;
    const uint circularShift = WaveGetLaneIndex() + laneMask & laneMask;
    uint waveReduction = 0;
    
    [unroll]
    for (uint i = WaveGetLaneIndex() + WavePartStart(gtid), k = 0;
        k < UINT4_PER_THREAD;
        i += WaveGetLaneCount(), ++k)
    {
        uint4 t = b_scan[i + PartStart(partIndex)];

        uint t2 = t.x;
        t.x += t.y;
        t.y = t2;

        t2 = t.x;
        t.x += t.z;
        t.z = t2;

        t2 = t.x;
        t.x += t.w;
        t.w = t2;
        
        const uint t3 = WaveReadLaneAt(t.x + WavePrefixSum(t.x), circularShift);
        g_shared[i] = SetXAddYZW((WaveGetLaneIndex() ? t3 : 0) + (k ? waveReduction : 0), t);
        waveReduction += WaveReadLaneAt(t3, 0);
    }
    
    if (!WaveGetLaneIndex())
        g_reduction[getWaveIndex(gtid)] = waveReduction;
}

inline void ScanExclusivePartial(uint gtid, uint partIndex)
{
    const uint laneMask = WaveGetLaneCount() - 1;
    const uint circularShift = WaveGetLaneIndex() + laneMask & laneMask;
    const uint finalPartSize = e_vectorizedSize - PartStart(partIndex);
    uint waveReduction = 0;
    
    [unroll]
    for (uint i = WaveGetLaneIndex() + WavePartStart(gtid), k = 0;
        k < UINT4_PER_THREAD;
        i += WaveGetLaneCount(), ++k)
    {
        uint4 t = i < finalPartSize ? b_scan[i + PartStart(partIndex)] : 0;

        uint t2 = t.x;
        t.x += t.y;
        t.y = t2;

        t2 = t.x;
        t.x += t.z;
        t.z = t2;

        t2 = t.x;
        t.x += t.w;
        t.w = t2;
        
        const uint t3 = WaveReadLaneAt(t.x + WavePrefixSum(t.x), circularShift);
        g_shared[i] = SetXAddYZW((WaveGetLaneIndex() ? t3 : 0) + (k ? waveReduction : 0), t);
        waveReduction += WaveReadLaneAt(t3, 0);
    }
    
    if (!WaveGetLaneIndex())
        g_reduction[getWaveIndex(gtid)] = waveReduction;
}

inline void ScanInclusiveFull(uint gtid, uint partIndex)
{
    const uint laneMask = WaveGetLaneCount() - 1;
    const uint circularShift = WaveGetLaneIndex() + laneMask & laneMask;
    uint waveReduction = 0;
    
    [unroll]
    for (uint i = WaveGetLaneIndex() + WavePartStart(gtid), k = 0;
        k < UINT4_PER_THREAD;
        i += WaveGetLaneCount(), ++k)
    {
        uint4 t = b_scan[i + PartStart(partIndex)];
        t.y += t.x;
        t.z += t.y;
        t.w += t.z;
        
        const uint t2 = WaveReadLaneAt(t.w + WavePrefixSum(t.w), circularShift);
        g_shared[i] = t + (WaveGetLaneIndex() ? t2 : 0) + (k ? waveReduction : 0);
        waveReduction += WaveReadLaneAt(t2, 0);
    }
    
    if (!WaveGetLaneIndex())
        g_reduction[getWaveIndex(gtid)] = waveReduction;
}

inline void ScanInclusivePartial(uint gtid, uint partIndex)
{
    const uint laneMask = WaveGetLaneCount() - 1;
    const uint circularShift = WaveGetLaneIndex() + laneMask & laneMask;
    const uint finalPartSize = e_vectorizedSize - PartStart(partIndex);
    uint waveReduction = 0;
    
    [unroll]
    for (uint i = WaveGetLaneIndex() + WavePartStart(gtid), k = 0;
        k < UINT4_PER_THREAD;
        i += WaveGetLaneCount(), ++k)
    {
        uint4 t = i < finalPartSize ? b_scan[i + PartStart(partIndex)] : 0;
        t.y += t.x;
        t.z += t.y;
        t.w += t.z;
        
        const uint t2 = WaveReadLaneAt(t.w + WavePrefixSum(t.w), circularShift);
        g_shared[i] = t + (WaveGetLaneIndex() ? t2 : 0) + (k ? waveReduction : 0);
        waveReduction += WaveReadLaneAt(t2, 0);
    }
    
    if (!WaveGetLaneIndex())
        g_reduction[getWaveIndex(gtid)] = waveReduction;
}

//Reduce the wave reductions
inline void LocalScanInclusiveWGE16(uint gtid, uint partIndex)
{
    if (gtid < BLOCK_DIM / WaveGetLaneCount())
        g_reduction[gtid] += WavePrefixSum(g_reduction[gtid]);
}

inline void LocalScanInclusiveWLT16(uint gtid, uint partIndex)
{
    const uint scanSize = BLOCK_DIM / WaveGetLaneCount();
    if (gtid < scanSize)
        g_reduction[gtid] += WavePrefixSum(g_reduction[gtid]);
    GroupMemoryBarrierWithGroupSync();
        
    const uint laneLog = countbits(WaveGetLaneCount() - 1);
    uint offset = laneLog;
    uint j = WaveGetLaneCount();
    for (; j < (scanSize >> 1); j <<= laneLog)
    {
        if (gtid < (scanSize >> offset))
        {
            g_reduction[((gtid + 1) << offset) - 1] +=
                WavePrefixSum(g_reduction[((gtid + 1) << offset) - 1]);
        }
        GroupMemoryBarrierWithGroupSync();
            
        if ((gtid & ((j << laneLog) - 1)) >= j && (gtid + 1) & (j - 1))
        {
            g_reduction[gtid] +=
                WaveReadLaneAt(g_reduction[((gtid >> offset) << offset) - 1], 0);
        }
        offset += laneLog;
    }
    GroupMemoryBarrierWithGroupSync();
        
    //If RADIX is not a power of lanecount
    const uint index = gtid + j;
    if (index < scanSize)
    {
        g_reduction[index] +=
            WaveReadLaneAt(g_reduction[((index >> offset) << offset) - 1], 0);
    }
}

//Pass in previous reductions, and write out
inline void DownSweepFull(uint gtid, uint partIndex, uint prevReduction)
{
    [unroll]
    for (uint i = WaveGetLaneIndex() + WavePartStart(gtid), k = 0;
        k < UINT4_PER_THREAD;
        i += WaveGetLaneCount(), ++k)
    {
        b_scan[i + PartStart(partIndex)] = g_shared[i] + prevReduction;
    }
}

inline void DownSweepPartial(uint gtid, uint partIndex, uint prevReduction)
{
    const uint finalPartSize = e_vectorizedSize - PartStart(partIndex);
    for (uint i = WaveGetLaneIndex() + WavePartStart(gtid), k = 0;
        k < UINT4_PER_THREAD && i < finalPartSize;
        i += WaveGetLaneCount(), ++k)
    {
        b_scan[i + PartStart(partIndex)] = g_shared[i] + prevReduction;
    }
}