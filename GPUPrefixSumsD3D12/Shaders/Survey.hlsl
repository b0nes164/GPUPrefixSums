/******************************************************************************
 * PrefixSumSurvey
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 3/5/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/

#define MAX_WAVE_SIZE   128U
#define GROUP_SIZE      256U

RWStructuredBuffer<uint> b_prefixSum :      register(u0);
RWStructuredBuffer<uint> b_validationInfo : register(u1);

groupshared uint g_shared[1024];

cbuffer cbPrefixSumSurvey : register(b0)
{
    uint e_size;
    uint padding0;
    uint padding1;
    uint padding2;
}

//sleep excess waves to accommodate all wave sizes
inline bool IsFirstWave(uint gtid)
{
    return gtid < WaveGetLaneCount();
}

//Pass some information to the validation kernel so
//it knows the size of the wave that performed the scan 
inline void WriteValidationInfo(uint gtid)
{
    if (!gtid)
        b_validationInfo[0] = WaveGetLaneCount();
}

inline void WriteValidationInfo(uint gtid, uint size)
{
    if (!gtid)
        b_validationInfo[0] = size;
}

/***********************************************************************************
* SERIAL prefix sums, using a single thread
************************************************************************************/
[numthreads(1, 1, 1)]
void SerialInclusive(uint3 gtid : SV_GroupThreadID)
{
    for (uint i = 1; i < e_size; ++i)
        b_prefixSum[i] += b_prefixSum[i - 1];
}

[numthreads(1, 1, 1)]
void SerialExclusive(uint3 gtid : SV_GroupThreadID)
{
    uint prev = 0;
    for (uint i = 0; i < e_size; ++i)
    {
        const uint t = b_prefixSum[i];
        b_prefixSum[i] = prev;
        prev += t;
    }
}

/***********************************************************************************
* WAVE-LEVEL sums of size EQUAL TO the WAVE size, using NO WAVE INTRINSICS
************************************************************************************/
[numthreads(MAX_WAVE_SIZE, 1, 1)]
void WaveKoggeStoneInclusive(uint3 gtid : SV_GroupThreadID)
{
    const uint end = WaveGetLaneCount() >> 1;
    for (uint i = 1; i <= end; i <<= 1)
    {
        uint t;
        if (WaveGetLaneIndex() >= i && IsFirstWave(gtid.x))    
            t = b_prefixSum[WaveGetLaneIndex() - i];
        DeviceMemoryBarrierWithGroupSync();
        
        if (WaveGetLaneIndex() >= i && IsFirstWave(gtid.x))
            b_prefixSum[WaveGetLaneIndex()] += t;
        DeviceMemoryBarrierWithGroupSync();
    }
    
    WriteValidationInfo(gtid.x);
}

[numthreads(MAX_WAVE_SIZE, 1, 1)]
void WaveKoggeStoneExclusive(uint3 gtid : SV_GroupThreadID)
{
    const uint end = WaveGetLaneCount() >> 1;
    for (uint i = 1; i <= end; i <<= 1)
    {
        uint t;
        if (WaveGetLaneIndex() >= i && IsFirstWave(gtid.x))    
            t = b_prefixSum[WaveGetLaneIndex() - i];
        DeviceMemoryBarrierWithGroupSync();
        
        if (WaveGetLaneIndex() >= i && IsFirstWave(gtid.x))
            b_prefixSum[WaveGetLaneIndex()] += t;
        DeviceMemoryBarrierWithGroupSync();
    }
        
    uint prev;
    if (WaveGetLaneIndex() && IsFirstWave(gtid.x))
        prev = b_prefixSum[WaveGetLaneIndex() - 1];
    DeviceMemoryBarrierWithGroupSync();
    
    if (IsFirstWave(gtid.x))
        b_prefixSum[WaveGetLaneIndex()] = WaveGetLaneIndex() ? prev : 0;
    
    WriteValidationInfo(gtid.x);
}

/***********************************************************************************
* WAVE-LEVEL sums of size EQUAL TO the WAVE size, using SHUFFLING
************************************************************************************/
[numthreads(MAX_WAVE_SIZE, 1, 1)]
void WaveKoggeStoneShuffleInclusive(uint3 gtid : SV_GroupThreadID)
{
    if (gtid.x < WaveGetLaneCount())
    {
        uint val = b_prefixSum[WaveGetLaneIndex()];
        
        const uint end = WaveGetLaneCount() >> 1;
        for (uint i = 1; i <= end; i <<= 1)
        {
            const uint t = WaveReadLaneAt(val, WaveGetLaneIndex() - i);
            if (WaveGetLaneIndex() >= i)
                val += t;
        }
        
        b_prefixSum[WaveGetLaneIndex()] = val;
    }
    
    WriteValidationInfo(gtid.x);
}

[numthreads(MAX_WAVE_SIZE, 1, 1)]
void WaveKoggeStoneShuffleExclusive(uint3 gtid : SV_GroupThreadID)
{
    //sleep excess threads to accommodate all wave sizes
    if (gtid.x < WaveGetLaneCount())
    {
        uint val = b_prefixSum[WaveGetLaneIndex()];
        
        const uint end = WaveGetLaneCount() >> 1;
        for (uint i = 1; i <= end; i <<= 1)
        {
            const uint t = WaveReadLaneAt(val, WaveGetLaneIndex() - i);
            if (WaveGetLaneIndex() >= i)
                val += t;
        }
        
        const uint shuffleUp = WaveReadLaneAt(val, WaveGetLaneIndex() - 1);
        b_prefixSum[WaveGetLaneIndex()] = WaveGetLaneIndex() ? shuffleUp : 0;
    }
    
    WriteValidationInfo(gtid.x);
}

/***********************************************************************************
* WAVE-LEVEL sums of size EQUAL TO the WAVE size, using PREFIX SUM INTRINSIC
************************************************************************************/
[numthreads(MAX_WAVE_SIZE, 1, 1)]
void WaveKoggeStoneIntrinsicInclusive(uint3 gtid : SV_GroupThreadID)
{
    if (IsFirstWave(gtid.x))
        b_prefixSum[WaveGetLaneIndex()] += WavePrefixSum(b_prefixSum[WaveGetLaneIndex()]);
    
    WriteValidationInfo(gtid.x);
}

[numthreads(MAX_WAVE_SIZE, 1, 1)]
void WaveKoggeStoneIntrinsicExclusive(uint3 gtid : SV_GroupThreadID)
{
    if (IsFirstWave(gtid.x))
        b_prefixSum[WaveGetLaneIndex()] = WavePrefixSum(b_prefixSum[WaveGetLaneIndex()]);
    
    WriteValidationInfo(gtid.x);
}

/***********************************************************************************
* WAVE-LEVEL sums of size GREATER THAN OR EQUAL TO the WAVE size, using PREFIX SUM INTRINSIC
************************************************************************************/
[numthreads(MAX_WAVE_SIZE, 1, 1)]
void WaveRakingReduceInclusive(uint3 gtid : SV_GroupThreadID)
{
    if (IsFirstWave(gtid.x))
    {
        uint prevReduction = 0;
        const uint highestLane = WaveGetLaneCount() - 1;
        
        for (uint i = WaveGetLaneIndex(); i < e_size; i += WaveGetLaneCount())
        {
            uint t = b_prefixSum[i];
            t += WavePrefixSum(t) + prevReduction;
            b_prefixSum[i] = t;
            prevReduction = WaveReadLaneAt(t, highestLane);
        }
    }
}

[numthreads(MAX_WAVE_SIZE, 1, 1)]
void WaveRakingReduceExclusive(uint3 gtid : SV_GroupThreadID)
{
    if (IsFirstWave(gtid.x))
    {
        uint prevReduction = 0;
        const uint laneMask = WaveGetLaneCount() - 1;
        const uint circularLaneShift = WaveGetLaneIndex() + laneMask & laneMask;
        
        for (uint i = WaveGetLaneIndex(); i < e_size; i += WaveGetLaneCount())
        {
            uint t = b_prefixSum[i];
            t += WavePrefixSum(t);
            t = WaveReadLaneAt(t, circularLaneShift);
            b_prefixSum[i] = (WaveGetLaneIndex() ? t : 0) + prevReduction;
            prevReduction += WaveReadLaneAt(t, 0);
        }
    }
}

/***********************************************************************************
* Block-level sums of size EQUAL TO to the GROUP size, using NO WAVE INTRINSICS
************************************************************************************/
[numthreads(GROUP_SIZE, 1, 1)]
void BlockKoggeStoneInclusive(uint3 gtid : SV_GroupThreadID)
{
    const uint end = GROUP_SIZE >> 1;
    for (uint i = 1; i <= end; i <<= 1)
    {
        uint t;
        if (gtid.x >= i)    
            t = b_prefixSum[gtid.x - i];
        DeviceMemoryBarrierWithGroupSync();
        
        if (gtid.x >= i)
            b_prefixSum[gtid.x] += t;
        DeviceMemoryBarrierWithGroupSync();
    }
}

[numthreads(GROUP_SIZE, 1, 1)]
void BlockKoggeStoneExclusive(uint3 gtid : SV_GroupThreadID)
{
    const uint end = GROUP_SIZE >> 1;
    for (uint i = 1; i <= end; i <<= 1)
    {
        uint t;
        if (gtid.x >= i)    
            t = b_prefixSum[gtid.x - i];
        DeviceMemoryBarrierWithGroupSync();
        
        if (gtid.x >= i)
            b_prefixSum[gtid.x] += t;
        DeviceMemoryBarrierWithGroupSync();
    }
        
    uint prev;
    if (gtid.x)
        prev = b_prefixSum[gtid.x - 1];
    DeviceMemoryBarrierWithGroupSync();
    b_prefixSum[gtid.x] = gtid.x ? prev : 0;
}

[numthreads(GROUP_SIZE, 1, 1)]
void BlockSklanskyInclusive(uint3 gtid : SV_GroupThreadID)
{
    uint offset = 0;
    for (uint i = 1; i < e_size; i <<= 1)
    {
        uint t;
        if (gtid.x & i)
            t = b_prefixSum[((gtid.x >> offset) << offset) - 1];
        DeviceMemoryBarrierWithGroupSync();
        
        if (gtid.x & i)
            b_prefixSum[gtid.x] += t;
        DeviceMemoryBarrierWithGroupSync();
        ++offset;
    }
}

[numthreads(GROUP_SIZE, 1, 1)]
void BlockSklanskyExclusive(uint3 gtid : SV_GroupThreadID)
{
    uint offset = 0;
    for (uint i = 1; i < e_size; i <<= 1)
    {
        uint t;
        if (gtid.x & i)
            t = b_prefixSum[((gtid.x >> offset) << offset) - 1];
        DeviceMemoryBarrierWithGroupSync();
        
        if (gtid.x & i)
            b_prefixSum[gtid.x] += t;
        DeviceMemoryBarrierWithGroupSync();
        ++offset;
    }
    
    uint prev;
    if (gtid.x)
        prev = b_prefixSum[gtid.x - 1];
    DeviceMemoryBarrierWithGroupSync();
    b_prefixSum[gtid.x] = gtid.x ? prev : 0;
}

//the classic
[numthreads(GROUP_SIZE, 1, 1)]
void BlockBrentKungBlellochInclusive(int3 gtid : SV_GroupThreadID)
{
    //Upsweep
    if (gtid.x < (e_size >> 1))
        b_prefixSum[(gtid.x << 1) + 1] += b_prefixSum[gtid.x << 1];
    
    uint offset = 1;
    for (uint j = e_size >> 2; j > 0; j >>= 1)
    {
        DeviceMemoryBarrierWithGroupSync();
        if (gtid.x < j)
            b_prefixSum[(((gtid.x << 1) + 2) << offset) - 1] += b_prefixSum[(((gtid.x << 1) + 1) << offset) - 1];
        ++offset;
    }
    
    //Downsweep
    for (uint j = 1; j < e_size; j <<= 1)
    {
        --offset;
        DeviceMemoryBarrierWithGroupSync();
        if (gtid.x < j - 1)
            b_prefixSum[(((gtid.x << 1) + 3) << offset) - 1] += b_prefixSum[(((gtid.x << 1) + 2) << offset) - 1];
    }
}

[numthreads(GROUP_SIZE, 1, 1)]
void BlockBrentKungBlellochExclusive(int3 gtid : SV_GroupThreadID)
{
    //Upsweep
    if (gtid.x < (e_size >> 1))
        b_prefixSum[(gtid.x << 1) + 1] += b_prefixSum[gtid.x << 1];
    
    uint offset = 1;
    for (uint j = e_size >> 2; j > 0; j >>= 1)
    {
        DeviceMemoryBarrierWithGroupSync();
        if (gtid.x < j)
            b_prefixSum[(((gtid.x << 1) + 2) << offset) - 1] += b_prefixSum[(((gtid.x << 1) + 1) << offset) - 1];
        ++offset;
    }
    
    if(!gtid.x)
        b_prefixSum[e_size - 1] = 0;
    
    //Downsweep
    for (uint j = 1; j < e_size; j <<= 1)
    {
        --offset;
        DeviceMemoryBarrierWithGroupSync();
        if (gtid.x < j)
        {
            const uint index1 = (((gtid.x << 1) + 1) << offset) - 1;
            const uint index2 = (((gtid.x << 1) + 2) << offset) - 1;
            const uint temp = b_prefixSum[index1];
            b_prefixSum[index1] = b_prefixSum[index2];
            b_prefixSum[index2] += temp;
        }
    }
}

//Mock multi level scan using shared memory to hold the intermediates
[numthreads(GROUP_SIZE, 1, 1)]
void BlockReduceScanInclusive(uint3 gtid : SV_GroupThreadID)
{
    const uint tileSize = 16; //For this demo, this must be a power of two
    const uint tileMask = tileSize - 1;
    
    //Reduce the tiles
    if (gtid.x < (e_size >> 1))
        b_prefixSum[(gtid.x << 1) + 1] += b_prefixSum[gtid.x << 1];
    DeviceMemoryBarrierWithGroupSync();
        
    uint mainOffset = 1;
    for (uint j = e_size >> 2; j >= tileSize; j >>= 1)
    {
        if(gtid.x < j)
            b_prefixSum[(((gtid.x << 1) + 2) << mainOffset) - 1] += b_prefixSum[(((gtid.x << 1) + 1) << mainOffset) - 1];
        DeviceMemoryBarrierWithGroupSync();
        ++mainOffset;
    }
    
    //Pass intermediates into secondary buffer
    if ((gtid.x & tileMask) == tileMask)
        g_shared[gtid.x / tileSize] = b_prefixSum[gtid.x];
    AllMemoryBarrierWithGroupSync();
    
    //Inclusive scan on the intermediates
    uint intermediateOffset = 0;
    const uint reductionSize = e_size / tileSize;
    for (uint j = 1; j < reductionSize; j <<= 1)
    {
        uint t;
        if ((gtid.x & j) && gtid.x < reductionSize)
            t = g_shared[((gtid.x >> intermediateOffset) << intermediateOffset) - 1];
        GroupMemoryBarrierWithGroupSync();
        
        if ((gtid.x & j) && gtid.x < reductionSize)
            g_shared[gtid.x] += t;
        GroupMemoryBarrierWithGroupSync();
        ++intermediateOffset;
    }
    
    //Pass in the intermediates
    --mainOffset;
    const uint index = (((gtid.x << 1) + 2) << mainOffset) + (1 << mainOffset + 1) - 1;
    if(index < e_size)
        b_prefixSum[index] += g_shared[index / tileSize - 1];
    
    //Downsweep
    for (uint j = tileSize; j < e_size; j <<= 1)
    {
        DeviceMemoryBarrierWithGroupSync();
        if (gtid.x < j)
            b_prefixSum[(((gtid.x << 1) + 3) << mainOffset) - 1] += b_prefixSum[(((gtid.x << 1) + 2) << mainOffset) - 1];
        --mainOffset;
    }
}

[numthreads(GROUP_SIZE, 1, 1)]
void BlockReduceScanExclusive(uint3 gtid : SV_GroupThreadID)
{
    const uint tileSize = 16; //For this demo, this must be a power of two
    const uint tileMask = tileSize - 1;
    
    //Reduce the tiles
    if (gtid.x < (e_size >> 1))
        b_prefixSum[(gtid.x << 1) + 1] += b_prefixSum[gtid.x << 1];
    DeviceMemoryBarrierWithGroupSync();
        
    uint mainOffset = 1;
    for (uint j = e_size >> 2; j >= tileSize; j >>= 1)
    {
        if (gtid.x < j)
            b_prefixSum[(((gtid.x << 1) + 2) << mainOffset) - 1] += b_prefixSum[(((gtid.x << 1) + 1) << mainOffset) - 1];
        DeviceMemoryBarrierWithGroupSync();
        ++mainOffset;
    }
    
    //Pass intermediates into secondary buffer
    if ((gtid.x & tileMask) == tileMask)
        g_shared[gtid.x / tileSize] = b_prefixSum[gtid.x];
    AllMemoryBarrierWithGroupSync();
    
    //Inclusive scan on the intermediates
    uint intermediateOffset = 0;
    const uint reductionSize = e_size / tileSize;
    for (uint j = 1; j < reductionSize; j <<= 1)
    {
        uint t;
        if ((gtid.x & j) && gtid.x < reductionSize)
            t = g_shared[((gtid.x >> intermediateOffset) << intermediateOffset) - 1];
        GroupMemoryBarrierWithGroupSync();
        
        if ((gtid.x & j) && gtid.x < reductionSize)
            g_shared[gtid.x] += t;
        GroupMemoryBarrierWithGroupSync();
        ++intermediateOffset;
    }
    
    //Pass in the intermediates
    --mainOffset;
    const uint index = (((gtid.x << 1) + 2) << mainOffset) - 1;
    if (index < e_size)
        b_prefixSum[index] = g_shared[index / tileSize - 1];
    DeviceMemoryBarrierWithGroupSync();
    
    //Downsweep
    for (uint j = tileSize; j < e_size; j <<= 1)
    {
        DeviceMemoryBarrierWithGroupSync();
        if (gtid.x < j)
        {
            const uint index1 = (((gtid.x << 1) + 1) << mainOffset) - 1;
            const uint index2 = (((gtid.x << 1) + 2) << mainOffset) - 1;
            const uint temp = b_prefixSum[index1];
            b_prefixSum[index1] = b_prefixSum[index2];
            b_prefixSum[index2] += temp;
        }
        --mainOffset;
    }
}

/***********************************************************************************
* Block-level sums of size EQUAL TO to the GROUP size, using WAVE INTRINSICS
************************************************************************************/
[numthreads(GROUP_SIZE, 1, 1)]
void BlockBrentKungIntrinsicInclusive(uint3 gtid : SV_GroupThreadID)
{
    //Upsweep
    //Warp-sized-radix KoggeStone embedded into BrentKung
    uint offset = 0;
    const uint laneLog = countbits(WaveGetLaneCount() - 1);
    for (uint j = e_size; j > 1; j >>= laneLog)
    {
        if (gtid.x < j)
            b_prefixSum[(gtid.x + 1 << offset) - 1] += WavePrefixSum(b_prefixSum[(gtid.x + 1 << offset) - 1]);
        DeviceMemoryBarrierWithGroupSync();
        offset += laneLog;
    }
    
    //Downsweep
    //Warp-sized radix propogation fans
    offset = laneLog;
    for (uint j = 1 << laneLog; j < e_size; j <<= laneLog)
    {
        if ((gtid.x & (j << laneLog) - 1) >= j && (gtid.x + 1 & j - 1))
            b_prefixSum[gtid.x] += b_prefixSum[((gtid.x >> offset) << offset) - 1];
        DeviceMemoryBarrierWithGroupSync();
        offset += laneLog;
    }
}

//Note, this only works when the inputSize <= WaveGetLaneCount() * WaveGetLaneCount()
//At the current input, will fail on WaveGetLaneCount() 4 and 8
[numthreads(GROUP_SIZE, 1, 1)]
void BlockBrentKungIntrinsicExclusive(uint3 gtid : SV_GroupThreadID)
{
    //Upsweep
    //Warp-sized-radix KoggeStone embedded into BrentKung
    uint offset = 0;
    const uint laneLog = countbits(WaveGetLaneCount() - 1);
    for (uint j = e_size; j > 1; j >>= laneLog)
    {
        if (gtid.x < j)
            b_prefixSum[(gtid.x + 1 << offset) - 1] += WavePrefixSum(b_prefixSum[(gtid.x + 1 << offset) - 1]);
        DeviceMemoryBarrierWithGroupSync();
        offset += laneLog;
    }
    
    //Downsweep
    //Warp-sized radix propogation fans
    for (uint j = 1 << offset; j >= WaveGetLaneCount(); j >>= laneLog)
    {
        uint prev;
        uint shuffleUp;
        if ((gtid.x & (j << laneLog) - 1) >= j && (gtid.x >> offset))
        {
            prev = b_prefixSum[((gtid.x >> offset) << offset) - 1];
            
            if (gtid.x < (j << laneLog))
                shuffleUp = ((gtid.x & (j - 1)) ? b_prefixSum[gtid.x - 1] : 0) + prev;
        }
        DeviceMemoryBarrierWithGroupSync();
        
        if ((gtid.x & (j << laneLog) - 1) >= j && (gtid.x >> offset))
        {
            if (gtid.x < (j << laneLog))
            {
                b_prefixSum[gtid.x] = shuffleUp;
            }
            else
            {
                if ((gtid.x + 1) & (j - 1))
                    b_prefixSum[gtid.x] += prev;
            }
        }
        offset -= laneLog;
    }
    
    uint prev;
    if (gtid.x < WaveGetLaneCount())
        prev = gtid.x ? b_prefixSum[gtid.x - 1] : 0;
    DeviceMemoryBarrierWithGroupSync();
    
    if (gtid.x < WaveGetLaneCount())
        b_prefixSum[gtid.x] = prev;
}

//Fuse the upsweep and downsweep
[numthreads(GROUP_SIZE, 1, 1)]
void BlockBrentKungFusedIntrinsicInclusive(uint3 gtid : SV_GroupThreadID)
{
    uint offset = 0;
    const uint laneLog = countbits(WaveGetLaneCount() - 1);
    for (uint j = 1; j < e_size; j <<= laneLog)
    {
        if (gtid.x < (e_size >> offset))
            b_prefixSum[(gtid.x + 1 << offset) - 1] += WavePrefixSum(b_prefixSum[(gtid.x + 1 << offset) - 1]);
        DeviceMemoryBarrierWithGroupSync();
        
        if ((gtid.x & (j << laneLog) - 1) >= j && (gtid.x + 1 & j - 1))
            b_prefixSum[gtid.x] += b_prefixSum[((gtid.x >> offset) << offset) - 1];
        offset += laneLog;
    }
}

//BlockBrentKungFusedIntrinsicExclusive will be covered in the shared memory 
//implementations section

[numthreads(GROUP_SIZE, 1, 1)]
void BlockSklanskyIntrinsicInclusive(uint3 gtid : SV_GroupThreadID)
{
    //Warp-sized radix Kogge-Stone
    b_prefixSum[gtid.x] += WavePrefixSum(b_prefixSum[gtid.x]);
    DeviceMemoryBarrierWithGroupSync();
    
    //Warp-sized radix Sklansky propogation fan
    const uint laneLog = countbits(WaveGetLaneCount() - 1);
    uint offset = laneLog;
    for (uint j = 1 << laneLog; j < e_size; j <<= 1) //Note that Sklansky increments by 1, not the size of the wave
    {
        const uint index = gtid.x + j;
        if (index < e_size && (index & j))
            b_prefixSum[index] += b_prefixSum[((index >> offset) << offset) - 1];
        DeviceMemoryBarrierWithGroupSync();
        ++offset;
    }
}

//An alternative implementaion, instead of sleeping threads when we reach an
//invalid index, we calculate the correct index
[numthreads(GROUP_SIZE, 1, 1)]
void BlockSklanskyIntrinsicInclusiveAlt(uint3 gtid : SV_GroupThreadID)
{
    //Warp-sized radix Kogge-Stone
    b_prefixSum[gtid.x] += WavePrefixSum(b_prefixSum[gtid.x]);
    DeviceMemoryBarrierWithGroupSync();
    
    const uint laneLog = countbits(WaveGetLaneCount() - 1);
    uint offset = laneLog;
    for (uint j = 1 << laneLog; j < e_size; j <<= 1) //Note that Sklansky increments by 1, not the size of the wave
    {
        const uint index = ((((gtid.x >> offset) << 1) + 1) << offset) + (gtid.x & (1 << offset) - 1);
        if (index < e_size)
            b_prefixSum[index] += b_prefixSum[((index >> offset) << offset) - 1];
        DeviceMemoryBarrierWithGroupSync();
        ++offset;
    }
}

[numthreads(GROUP_SIZE, 1, 1)]
void BlockSklanskyIntrinsicExclusive(uint3 gtid : SV_GroupThreadID)
{
    //Warp-sized radix Kogge-Stone
    const uint laneMask = WaveGetLaneCount() - 1;
    const uint circularLaneShift = WaveGetLaneIndex() + 1 & laneMask;
    
    uint t = b_prefixSum[gtid.x];
    t += WavePrefixSum(t);
    b_prefixSum[circularLaneShift + (gtid.x & ~laneMask)] = t;
    DeviceMemoryBarrierWithGroupSync();
    
    //Warp-sized radix Sklansky propogation fan
    const uint laneLog = countbits(laneMask);
    uint offset = laneLog;
    uint prev;
    if (gtid.x >= WaveGetLaneCount() && (gtid.x & WaveGetLaneCount()))
        prev = b_prefixSum[((gtid.x >> offset) << offset) - WaveGetLaneCount()];
    DeviceMemoryBarrierWithGroupSync();
    
    if (gtid.x >= WaveGetLaneCount() && (gtid.x & WaveGetLaneCount()))
        b_prefixSum[gtid.x] += prev;
    ++offset;
    
    for (uint j = 1 << laneLog + 1; j <= e_size; j <<= 1) //Note that Sklansky increments by 1, not the size of the wave
    {
        if (gtid.x >= j && (gtid.x & j))
            prev = b_prefixSum[((gtid.x >> offset) << offset) - WaveGetLaneCount()];
        DeviceMemoryBarrierWithGroupSync();

        if (gtid.x >= j && (gtid.x & j))
            b_prefixSum[gtid.x] += prev;
        
        ++offset;
    }
    
    if (!(gtid.x & laneMask))
        g_shared[gtid.x / WaveGetLaneCount()] = b_prefixSum[gtid.x];
    GroupMemoryBarrierWithGroupSync();
    
    if (!(gtid.x & laneMask))
        b_prefixSum[gtid.x] = gtid.x / WaveGetLaneCount() ? g_shared[gtid.x / WaveGetLaneCount() - 1] : 0;
}

//Requires GROUP_SIZE / WaveGetLaneCount() <= WaveGetLaneCount()
//At the current input and GROUP_SIZE, will fail on WaveGetLaneCount() 4 and 8
[numthreads(GROUP_SIZE, 1, 1)]
void BlockRakingReduceIntrinsicInclusive(uint3 gtid : SV_GroupThreadID)
{
    b_prefixSum[gtid.x] += WavePrefixSum(b_prefixSum[gtid.x]);
    DeviceMemoryBarrierWithGroupSync();
    
    if (gtid.x < e_size / WaveGetLaneCount())
        b_prefixSum[(gtid.x + 1) * WaveGetLaneCount() - 1] += WavePrefixSum(b_prefixSum[(gtid.x + 1) * WaveGetLaneCount() - 1]);
    DeviceMemoryBarrierWithGroupSync();
    
    const uint highestLane = WaveGetLaneCount() - 1;
    if (WaveGetLaneIndex() != highestLane)
        b_prefixSum[gtid.x] += gtid.x / WaveGetLaneCount() ? WaveReadLaneAt(b_prefixSum[gtid.x - 1], 0) : 0;
}

[numthreads(GROUP_SIZE, 1, 1)]
void BlockRakingReduceIntrinsicExclusive(uint3 gtid : SV_GroupThreadID)
{
    const uint laneMask = WaveGetLaneCount() - 1;
    const uint circularLaneShift = WaveGetLaneIndex() + laneMask & laneMask;

    uint t = b_prefixSum[gtid.x];
    t += WavePrefixSum(t);
    b_prefixSum[gtid.x] = WaveReadLaneAt(t, circularLaneShift);
    DeviceMemoryBarrierWithGroupSync();
    
    if (gtid.x < e_size / WaveGetLaneCount())
        b_prefixSum[gtid.x * WaveGetLaneCount()] = WavePrefixSum(b_prefixSum[gtid.x * WaveGetLaneCount()]);
    DeviceMemoryBarrierWithGroupSync();
    
    if(WaveGetLaneIndex())
        b_prefixSum[gtid.x] += WaveReadLaneAt(b_prefixSum[gtid.x - 1], 1);
}

/***********************************************************************************
* Block-level sums of size EQUAL TO to the GROUP size, using WAVE INTRINSICS and SHARED memory
************************************************************************************/
[numthreads(GROUP_SIZE, 1, 1)]
void SharedBrentKungFusedIntrinsicInclusive(uint3 gtid : SV_GroupThreadID)
{
    g_shared[gtid.x] = b_prefixSum[gtid.x];
    g_shared[gtid.x] += WavePrefixSum(g_shared[gtid.x]);
    
    if (gtid.x < WaveGetLaneCount())
        b_prefixSum[gtid.x] = g_shared[gtid.x];
    GroupMemoryBarrierWithGroupSync();
    
    const uint laneLog = countbits(WaveGetLaneCount() - 1);
    uint offset = laneLog;
    uint j = WaveGetLaneCount();
    for (; j < (e_size >> 1); j <<= laneLog)
    {
        if (gtid.x < (e_size >> offset))
        {
            g_shared[(gtid.x + 1 << offset) - 1] +=
                WavePrefixSum(g_shared[(gtid.x + 1 << offset) - 1]);
        }
        GroupMemoryBarrierWithGroupSync();
        
        if ((gtid.x & (j << laneLog) - 1) >= j)
        {
            if (gtid.x < (j << laneLog))
            {
                //Write out, avoid an unecessary store into shared memory
                b_prefixSum[gtid.x] = g_shared[gtid.x] + ((gtid.x + 1) & (j - 1) ?
                    WaveReadLaneAt(g_shared[((gtid.x >> offset) << offset) - 1], 0) : 0);
            }
            else
            {
                //Fan
                if ((gtid.x + 1 & j - 1))
                    g_shared[gtid.x] += WaveReadLaneAt(g_shared[((gtid.x >> offset) << offset) - 1], 0);
            }
        }
        offset += laneLog;
    }
    GroupMemoryBarrierWithGroupSync();
    
    //If inputSize is not a power of WaveGetLaneCount()
    const uint index = gtid.x + j;
    if (index < e_size)
    {
        b_prefixSum[index] = g_shared[index] + ((index + 1) & (j - 1) ?
            WaveReadLaneAt(g_shared[((index >> offset) << offset) - 1], 0) : 0);
    }
}

[numthreads(GROUP_SIZE, 1, 1)]
void SharedBrentKungFusedIntrinsicExclusive(uint3 gtid : SV_GroupThreadID)
{
    g_shared[gtid.x] = b_prefixSum[gtid.x];
    g_shared[gtid.x] += WavePrefixSum(g_shared[gtid.x]);
    GroupMemoryBarrierWithGroupSync();
    
    const uint laneLog = countbits(WaveGetLaneCount() - 1);
    uint offset = laneLog;
    uint j = WaveGetLaneCount();
    for (; j < (e_size >> 1); j <<= laneLog)
    {
        if (gtid.x < (e_size >> offset))
        {
            g_shared[(gtid.x + 1 << offset) - 1] +=
                WavePrefixSum(g_shared[(gtid.x + 1 << offset) - 1]);
        }
        GroupMemoryBarrierWithGroupSync();
        
        if ((gtid.x & (j << laneLog) - 1) >= j)
        {
            if (gtid.x < (j << laneLog))
            {
                //Write out, avoid an unecessary store into shared memory
                b_prefixSum[gtid.x] = ((gtid.x & (j - 1)) ? g_shared[gtid.x - 1] : 0) +
                    WaveReadLaneAt(g_shared[((gtid.x >> offset) << offset) - 1], 0);
            }
            else
            {
                //Fan
                if ((gtid.x + 1 & j - 1))
                    g_shared[gtid.x] += WaveReadLaneAt(g_shared[((gtid.x >> offset) << offset) - 1], 0);
            }
        }
        offset += laneLog;
    }
    GroupMemoryBarrierWithGroupSync();
    
    //If inputSize is not a power of WaveGetLaneCount()
    const uint index = gtid.x + j;
    if (index < e_size)
    {
        b_prefixSum[index] = ((index & (j - 1)) ? g_shared[index - 1] : 0) +
            WaveReadLaneAt(g_shared[((index >> offset) << offset) - 1], 0);
    }
    
    if (gtid.x < WaveGetLaneCount())
        b_prefixSum[gtid.x] = gtid.x ? g_shared[gtid.x - 1] : 0;
}

//Requires GROUP_SIZE / WaveGetLaneCount() <= WaveGetLaneCount()
//At the current input and GROUP_SIZE, will fail on WaveGetLaneCount() 4 and 8
[numthreads(GROUP_SIZE, 1, 1)]
void SharedRakingReduceIntrinsicInclusive(uint3 gtid : SV_GroupThreadID)
{
    g_shared[gtid.x] = b_prefixSum[gtid.x];
    g_shared[gtid.x] += WavePrefixSum(g_shared[gtid.x]);
    GroupMemoryBarrierWithGroupSync();
    
    if (gtid.x < e_size / WaveGetLaneCount())
        g_shared[(gtid.x + 1) * WaveGetLaneCount() - 1] += WavePrefixSum(g_shared[(gtid.x + 1) * WaveGetLaneCount() - 1]);
    GroupMemoryBarrierWithGroupSync();
    
    const uint highestLane = WaveGetLaneCount() - 1;
    b_prefixSum[gtid.x] = g_shared[gtid.x] + ((WaveGetLaneIndex() != highestLane && gtid.x / WaveGetLaneCount()) ?
        WaveReadLaneAt(g_shared[gtid.x - 1], 0) : 0);
}

[numthreads(GROUP_SIZE, 1, 1)]
void SharedRakingReduceIntrinsicExclusive(uint3 gtid : SV_GroupThreadID)
{
    const uint laneMask = WaveGetLaneCount() - 1;
    const uint circularLaneShift = WaveGetLaneIndex() + 1 & laneMask;
    
    const uint t = b_prefixSum[gtid.x];;
    g_shared[circularLaneShift + (gtid.x & ~laneMask)] = t + WavePrefixSum(t);
    GroupMemoryBarrierWithGroupSync();

    if (gtid.x < e_size / WaveGetLaneCount())
        g_shared[gtid.x * WaveGetLaneCount()] = WavePrefixSum(g_shared[gtid.x * WaveGetLaneCount()]);
    GroupMemoryBarrierWithGroupSync();
    
    b_prefixSum[gtid.x] = g_shared[gtid.x] + (WaveGetLaneIndex() ?
        WaveReadLaneAt(g_shared[gtid.x - 1], 1) : 0);
}

/***********************************************************************************
* True block level sum, incorporating all previous techniques create a wave size 
* agnostic prefix sum that can accomodate any input size.
************************************************************************************/
#define VAL_PER_THREAD  4
#define PARTITION_SIZE  1024
groupshared uint g_reduction[64];

inline uint getWaveIndex(uint gtid)
{
    return gtid / WaveGetLaneCount();
}

inline uint WavePartStart(uint gtid)
{
    return WaveGetLaneCount() * VAL_PER_THREAD *
        getWaveIndex(gtid);
}

inline uint PartStart(uint partitionIndex)
{
    return partitionIndex * PARTITION_SIZE;
}

//Raking scan to perform the wave level prefix sum
inline void ScanInclusiveFull(uint gtid, uint partIndex)
{
    uint waveReduction = 0;
    const uint highestLane = WaveGetLaneCount() - 1;
    
    [unroll]
    for (uint i = WaveGetLaneIndex() + WavePartStart(gtid), k = 0;
        k < VAL_PER_THREAD;
        i += WaveGetLaneCount(), ++k)
    {
        uint t = b_prefixSum[i + PartStart(partIndex)];
        t += WavePrefixSum(t);
        g_shared[i] = t + waveReduction;
        waveReduction += WaveReadLaneAt(t, highestLane);
    }

    if (!WaveGetLaneIndex())
        g_reduction[getWaveIndex(gtid)] = waveReduction;
}

inline void ScanInclusivePartial(uint gtid, uint partIndex)
{
    uint waveReduction = 0;
    const uint highestLane = WaveGetLaneCount() - 1;
    const uint finalPartSize = e_size - PartStart(partIndex);
    
    [unroll]
    for (uint i = WaveGetLaneIndex() + WavePartStart(gtid), k = 0;
        k < VAL_PER_THREAD;
        i += WaveGetLaneCount(), ++k)
    {
        uint t = i < finalPartSize ? b_prefixSum[i + PartStart(partIndex)] : 0;
        t += WavePrefixSum(t);
        g_shared[i] = t + waveReduction;
        waveReduction += WaveReadLaneAt(t, highestLane);
    }

    if (!WaveGetLaneIndex())
        g_reduction[getWaveIndex(gtid)] = waveReduction;
}

inline void ScanExclusiveFull(uint gtid, uint partIndex)
{
    uint waveReduction = 0;
    const uint laneMask = WaveGetLaneCount() - 1;
    const uint circularLaneShift = WaveGetLaneIndex() + laneMask & laneMask;
    
    [unroll]
    for (uint i = WaveGetLaneIndex() + WavePartStart(gtid), k = 0;
        k < VAL_PER_THREAD;
        i += WaveGetLaneCount(), ++k)
    {
        uint t = b_prefixSum[i + PartStart(partIndex)];
        t = WaveReadLaneAt(t + WavePrefixSum(t), circularLaneShift);
        g_shared[i] = (WaveGetLaneIndex() ? t : 0) + waveReduction;
        waveReduction += WaveReadLaneAt(t, 0);
    }

    if (!WaveGetLaneIndex())
        g_reduction[getWaveIndex(gtid)] = waveReduction;
}

inline void ScanExclusivePartial(uint gtid, uint partIndex)
{
    uint waveReduction = 0;
    const uint laneMask = WaveGetLaneCount() - 1;
    const uint circularLaneShift = WaveGetLaneIndex() + laneMask & laneMask;
    const uint finalPartSize = e_size - PartStart(partIndex);
    
    [unroll]
    for (uint i = WaveGetLaneIndex() + WavePartStart(gtid), k = 0;
        k < VAL_PER_THREAD;
        i += WaveGetLaneCount(), ++k)
    {
        uint t = i < finalPartSize ? b_prefixSum[i + PartStart(partIndex)] : 0;
        t = WaveReadLaneAt(t + WavePrefixSum(t), circularLaneShift);
        g_shared[i] = (WaveGetLaneIndex() ? t : 0) + waveReduction;
        waveReduction += WaveReadLaneAt(t, 0);
    }

    if (!WaveGetLaneIndex())
        g_reduction[getWaveIndex(gtid)] = waveReduction;
}

//Scan over the reductions
inline void LocalScanWGE16(uint gtid)
{
    if (gtid < GROUP_SIZE / WaveGetLaneCount())
        g_reduction[gtid] += WavePrefixSum(g_reduction[gtid]);
}

inline void LocalScanWLT16(uint gtid)
{

}

inline void DownSweepFull(uint gtid, uint partIndex, uint prevReduction)
{
    [unroll]
    for (uint i = WaveGetLaneIndex() + WavePartStart(gtid), k = 0;
        k < VAL_PER_THREAD;
        i += WaveGetLaneCount(), ++k)
    {
        b_prefixSum[i + PartStart(partIndex)] = g_shared[i] + prevReduction;
    }
}

inline void DownSweepPartial(uint gtid, uint partIndex, uint prevReduction)
{
    const uint finalPartSize = e_size - PartStart(partIndex);
    for (uint i = WaveGetLaneIndex() + WavePartStart(gtid), k = 0;
        k < VAL_PER_THREAD && i < finalPartSize;
        i += WaveGetLaneCount(), ++k)
    {
        b_prefixSum[i + PartStart(partIndex)] = g_shared[i] + prevReduction;
    }
}

[numthreads(GROUP_SIZE, 1, 1)]
void TrueBlockInclusiveScan(uint3 gtid : SV_GroupThreadID)
{
    const uint partitions = (e_size + PARTITION_SIZE - 1) / PARTITION_SIZE - 1;
    
    uint reduction = 0;
    uint partitionIndex = 0;
    for (; partitionIndex < partitions; ++partitionIndex)
    {
        ScanInclusiveFull(gtid.x, partitionIndex);
        GroupMemoryBarrierWithGroupSync();
        
        if (WaveGetLaneCount() >= 16)
            LocalScanWGE16(gtid.x);
        
        if (WaveGetLaneCount() < 16)
            LocalScanWLT16(gtid.x);
        GroupMemoryBarrierWithGroupSync();
        
        const uint prevReduction = (getWaveIndex(gtid.x) ? g_reduction[getWaveIndex(gtid.x) - 1] : 0) + reduction;
        DownSweepFull(gtid.x, partitionIndex, prevReduction);
        
        reduction += WaveReadLaneAt(g_reduction[GROUP_SIZE / WaveGetLaneCount() - 1], 0);
        GroupMemoryBarrierWithGroupSync();
    }
    
    ScanInclusivePartial(gtid.x, partitionIndex);
    GroupMemoryBarrierWithGroupSync();
    
    if (WaveGetLaneCount() >= 16)
        LocalScanWGE16(gtid.x);
        
    if (WaveGetLaneCount() < 16)
        LocalScanWLT16(gtid.x);
    GroupMemoryBarrierWithGroupSync();
        
    const uint prevReduction = (getWaveIndex(gtid.x) ? g_reduction[getWaveIndex(gtid.x) - 1] : 0) + reduction;
    DownSweepPartial(gtid.x, partitionIndex, prevReduction);
}

[numthreads(GROUP_SIZE, 1, 1)]
void TrueBlockExclusiveScan(uint3 gtid : SV_GroupThreadID)
{
    const uint partitions = (e_size + PARTITION_SIZE - 1) / PARTITION_SIZE - 1;
    
    uint reduction = 0;
    uint partitionIndex = 0;
    for (; partitionIndex < partitions; ++partitionIndex)
    {
        ScanExclusiveFull(gtid.x, partitionIndex);
        GroupMemoryBarrierWithGroupSync();
        
        if (WaveGetLaneCount() >= 16)
            LocalScanWGE16(gtid.x);
        
        if (WaveGetLaneCount() < 16)
            LocalScanWLT16(gtid.x);
        GroupMemoryBarrierWithGroupSync();
        
        const uint prevReduction = (getWaveIndex(gtid.x) ? g_reduction[getWaveIndex(gtid.x) - 1] : 0) + reduction;
        DownSweepFull(gtid.x, partitionIndex, prevReduction);
        
        reduction += WaveReadLaneAt(g_reduction[GROUP_SIZE / WaveGetLaneCount() - 1], 0);
        GroupMemoryBarrierWithGroupSync();
    }
    
    ScanExclusivePartial(gtid.x, partitionIndex);
    GroupMemoryBarrierWithGroupSync();
    
    if (WaveGetLaneCount() >= 16)
        LocalScanWGE16(gtid.x);
        
    if (WaveGetLaneCount() < 16)
        LocalScanWLT16(gtid.x);
    GroupMemoryBarrierWithGroupSync();
        
    const uint prevReduction = (getWaveIndex(gtid.x) ? g_reduction[getWaveIndex(gtid.x) - 1] : 0) + reduction;
    DownSweepPartial(gtid.x, partitionIndex, prevReduction);
}