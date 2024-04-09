/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 3/6/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#define VAL_THREADS     256

RWStructuredBuffer<uint> b_scan             : register(u0);
RWStructuredBuffer<uint> b_scanValidation   : register(u1);
RWStructuredBuffer<uint> b_errorCount       : register(u2);

cbuffer cbPrefixSum : register(b0)
{
    uint e_size;
    uint e_threadBlocks;
    uint e_seed;
    uint padding0;
};

//Init all elements to one for fast validation
[numthreads(VAL_THREADS, 1, 1)]
void InitOne(int3 id : SV_DispatchThreadID)
{
    const uint size = e_size;
    const uint inc = VAL_THREADS * 256;
    for (uint i = id.x; i < size; i += inc)
        b_scan[i] = 1;
}

//randomized elements must be saved to be checked against
//Because of the numerical limit on the reduction,
//we limit both the values of the randomized value to 2^10 AND
//we limit the max size of the test to 2^20
//Hybrid Tausworthe
//GPU GEMS CH37 Lee Howes + David Thomas
#define TAUS_STEP_1 ((z1 & 4294967294U) << 12) ^ (((z1 << 13) ^ z1) >> 19)
#define TAUS_STEP_2 ((z2 & 4294967288U) << 4) ^ (((z2 << 2) ^ z2) >> 25)
#define TAUS_STEP_3 ((z3 & 4294967280U) << 17) ^ (((z3 << 3) ^ z3) >> 11)
#define LCG_STEP    (z4 * 1664525 + 1013904223U)
#define HYBRID_TAUS (z1 ^ z2 ^ z3 ^ z4)
[numthreads(VAL_THREADS, 1, 1)]
void InitRandom(uint3 id : SV_DispatchThreadID)
{
    const uint size = e_size;
    const uint inc = VAL_THREADS * 256;
    
    uint z1 = (id.x << 2) * e_seed;
    uint z2 = ((id.x << 2) + 1) * e_seed;
    uint z3 = ((id.x << 2) + 2) * e_seed;
    uint z4 = ((id.x << 2) + 3) * e_seed;
    
    z1 = TAUS_STEP_1;
    z2 = TAUS_STEP_2;
    z3 = TAUS_STEP_3;
    z4 = LCG_STEP;
    
    for (uint i = id.x; i < size; i += inc)
    {
        const uint t = HYBRID_TAUS & 1023;
        b_scan[i] = t;
        b_scanValidation[i] = t;
    }
}

[numthreads(1, 1, 1)]
void ClearErrorCount(uint3 id : SV_DispatchThreadID)
{
    b_errorCount[0] = 0;
}

//The correct prefix sum for a input initialized to one is its index
[numthreads(VAL_THREADS, 1, 1)]
void ValidateOneInclusive(uint3 id : SV_DispatchThreadID)
{
    const uint size = e_size;
    const uint inc = VAL_THREADS * 256;
    for (uint i = id.x; i < size; i += inc)
    {
        if (b_scan[i] != i + 1)
            InterlockedAdd(b_errorCount[0], 1);
    }
}

[numthreads(VAL_THREADS, 1, 1)]
void ValidateOneExclusive(uint3 id : SV_DispatchThreadID)
{
    const uint size = e_size;
    const uint inc = VAL_THREADS * 256;
    for (uint i = id.x; i < size; i += inc)
    {
        if (b_scan[i] != i)
            InterlockedAdd(b_errorCount[0], 1);
    }
}

//Randomized input is validated by a single thread
[numthreads(1, 1, 1)]
void ValidateRandomInclusive(uint3 id : SV_DispatchThreadID)
{
    const uint size = e_size;
    uint sum = 0;
    for (uint i = id.x; i < size; ++i)
    {
        sum += b_scanValidation[i];
        if (b_scan[i] != sum)
            InterlockedAdd(b_errorCount[0], 1);
    }
}

//Randomized input is validated by a single thread
[numthreads(1, 1, 1)]
void ValidateRandomExclusive(uint3 id : SV_DispatchThreadID)
{
    const uint size = e_size;
    uint sum = 0;
    for (uint i = id.x; i < size; ++i)
    {
        if (b_scan[i] != sum)
            InterlockedAdd(b_errorCount[0], 1);
        sum += b_scanValidation[i];
    }
}