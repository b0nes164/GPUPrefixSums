# GPU Prefix Sums
![Prefix Sum Speeds, in Unity Editor, RTX 2080 Super](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/7fd486be-cfd5-4a03-b24b-2d850431d8fd)

This project is a survey of GPU prefix sums, ranging from the warp to the device level, with the aim of providing developers an uncompiled look at modern prefix sum implementations. In particular, this project was inspired by Duane Merill's [research](https://research.nvidia.com/person/duane-merrill%2520iii) and includes implementations of Merill and Garland's [Chained Scan with Decoupled Lookback](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back), which is how we are able to reach speeds approaching `MemCopy()`. Finally, this project was written in HLSL for compute shaders, though with reasonable knowledge of GPU programming it is easily portable. 

**To the best of my knowledge, all algorithms included in this project are in the public domain and free to use, as is this project itself. (Chained Scan is licensed under BSD-2, and Blelloch's algorithm was released through GPU Gems. This is not legal advice.)** 
<!-- This content will not appear in the rendered Markdown -->
<!-- This content will not appear in the rendered Markdown -->
<!-- This content will not appear in the rendered Markdown -->
<!-- This content will not appear in the rendered Markdown -->
<!-- This content will not appear in the rendered Markdown -->
<!-- This content will not appear in the rendered Markdown -->
<!-- This content will not appear in the rendered Markdown -->
<!-- This content will not appear in the rendered Markdown -->

# Important Notes
<details>
  
  <summary>Currently, this project does not work on AMD or integrated graphics hardware.</summary>
  
</br>Unfortunately, AMD, Nvidia, and integrated graphics all have different wave sizes, which means that code that synchronizes threads on a wave level, like we do, must be manually tuned for each hardware case. Because we are manually unrolling loops with the `[unroll(x)]` attribute, changing the wave size also necessitates changing these unrolls. Furthermore Unity does not support runtime compilation of compute shaders so we cannot poll the hardware at runtime to compile a targetted shader variant. Although Unity does have the `multi_compile` functionality, it is a very cumbersome solution because it means maintaining and compiling a copy of each kernel for each hardware case.

Eventually I plan on making a tool that parse my Nvidia targetted shader to output a `multi_compile` version for all hardware cases, but until then non-Nvidia users will have to manually change the preprocessor macros and unrolls in the `.compute` file. To do so, open up the `.compute` file of the desired scan. Inside you will find the preprocessor macros like so:

  ![image](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/a1290a27-4106-4b3e-81d9-26a2e41bcca6)

  Comment out or delete the Nvidia values, and uncomment the AMD values. Next, search for the `[unroll(x)]` attributes and change the value to match the loop iterations at the new wave size:

  ![image](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/6e20a567-3df5-45ab-ab5b-448fbc5bb2a1)
 
</details>

<details>
  
  <summary>Chained Scan with Decoupled Lookback is not guaranteed to work on Nvidia cards older than Volta or AMD cards older than ????.</summary>

</br>Decoupled Lookback relies on two concepts to function properly: guaranteed forward progress of threads and fair scheduling of thread groups. This is because we effectively create threadblock level spinlocks during the lookback phase of the algorithm. Without these guaruntees, there is a chance that a threadblock never unlocks or that a threadblock whose depedent aggregate is already available is kept waiting for a suboptimal period of time. Thus, hardware models without these features may not see the same speedup or may fail to work altogether. If you wish to read more about the portability issues, and some of the general challenges of implementing chained decoupled scan, I would highly recommend reading Raph Levien’s [blog](https://raphlinus.github.io/gpu/2020/04/30/prefix-sum.html) detailing his experience with it. To read more on the issue of GPU workgroup progress models I recommend this [paper](https://arxiv.org/abs/2109.06132).

</details>

<details>
  
  <summary>Currently the maximum aggregate sum supported in Chained Scan with Decoupled Lookback is 2^30.</summary>

</br>In order to maintain coherency of the flag values between threadblocks, we have to bit-pack the threadblock aggregate into into the same value as the status flag. The flag value takes 2 bits, so we are left 30 bits for the aggregate. Although shader model 6.6 does support 64-bit values and atomics, enabling these features in Unity is difficult, and I will not include it until Unity moves the feature out of beta.
  
</details>

<details>
  
  <summary>DX12 is a must as well as a minimum Unity version of 2021.1 or later</summary>

</br>As we make heavy use of [WaveIntrinsics](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/hlsl-shader-model-6-0-features-for-direct3d-12), we need `pragma use_dxc` [to access shader model 6.0](https://forum.unity.com/threads/unity-is-adding-a-new-dxc-hlsl-compiler-backend-option.1086272/).

</details>

<details>
  
  <summary>All scans are inclusive.</summary>
  
</details>

<!-- This content will not appear in the rendered Markdown -->
<!-- This content will not appear in the rendered Markdown -->
<!-- This content will not appear in the rendered Markdown -->
<!-- This content will not appear in the rendered Markdown -->
<!-- This content will not appear in the rendered Markdown -->
<!-- This content will not appear in the rendered Markdown -->
<!-- This content will not appear in the rendered Markdown -->
<!-- This content will not appear in the rendered Markdown -->
<!-- This content will not appear in the rendered Markdown -->

# To Use This Project

1. Download or clone the repository.
2. Drag the contents of `src` into a desired folder within a Unity project.
3. Every scan variant has a compute shader and a dispatcher. Attach the desired scan's dispatcher to an empty game object. All scan dispatchers are named  `ScanNameHere + Dispatcher.cs`.
4. Attach the matching compute shader to the game object. All compute shaders are named `ScanNameHere.compute`. The dispatcher will return an error if you attach the wrong shader.
5. Ensure the sliders are set to nonzero values.

If you did this correctly you should see this in the inspector:


![image](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/70bb5097-fff2-44e5-b396-2930a059fbad)
<details>

<summary>

## Testing Suite

</summary>
  
![Tests](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/9d79a090-2f13-4031-925a-ef0788e75bf3)

Every scan dispatcher inherits a testing suite that can be controlled in the inspector.

+ `Validate Sum` performs `Kernel Iterations` number of prefix sums on a buffer of size 2^`SizeExponent`, then reads result back to host CPU memory to validate the results. If you have `Validate Text` ticked, any errors found will printed in the debug log. For very large sums, this can take several minutes, so if you don't want absolutely every error printed you can tick `Quick Text` which limits the number of errors printed to 1024.

+ `Validate Sum Random`/ `Validate Sum Monotonic` perform a single prefix sum on a buffer of either random values or a sequence of postive integers. By default, the buffer is filled with the value 1. This makes error checking very simple, because the corresponding correct prefix sum is the sequence of positive integers up to the size of the buffer. However this makes some other errors possible, so to cover our bases we include this test.

+ `Debug At Size` performs a single prefix sum on a buffer of size `Specific Size`. This is a way to directly test the validty of the prefix sum on buffer sizes that are not powers of two.

+ `Debug State` performs a single prefix sum of size 2^`SizeExponent`, then prints the contents of the `State Buffer` into the debug log. The `State Buffer` contains the threadblock aggregates in the device level scans, so `Debug State` can be used to verify that the aggregation is performed correctly.

+ `Torture Test` performs `Kernel Iterations` number of prefix sums at size 2^`SizeExponent`. It does not perform any validation and should be used to verify the stability of a scan. 

+ `Timing Test` performs `Kernel Iterations` number of prefix sums at size 2^`SizeExponent`, then prints the total time taken to complete each kernel in the debug log. However, **this is not** the actual speed of the algorithm. This is because, to the best of my knowledge, there is no way to time the kernel in-situ in HLSL. Neither is there a way to directly record kernel completion time in host CPU code, at least not in Unity. Instead we are forced to make an `AsyncGPUReadback.Request()`, which waits until the kernel completes writing to the buffer then reads **the entire buffer** back into host CPU memory. While this does time the kernel, the time produced will also include the GPU - CPU readback time, **which can be as much as 99% of the total time value. Thus, the time value produced by this test should only be used as a relative measurement between tests.** To see how the algorithm was actually timed, see the Testing Methodology section below.

+ `Record Timing Data` performs `Kernel Iterations` number of prefix sums at size 2^`SizeExponent`, then writes each indivual kernel completion time to a `.csv` file. Note that this has the same shortcomings as `Timing Test`.

+ `Validate Powers of Two` performs a single prefix sum at all powers of two which the prefix sum is designed for. Typically for the device-level scans this is 2^21 to 2^28, 2^28 being the largest sized buffer that Unity can allocate.

+ `Validate All Off Sizes` performs a series of tests to ensure that the perfix sum correctly handles non-powers-of-two buffer sizes. This test can take quite some time.

+ `Advanced Timing Mode` switches from the default kernel to a timing-specific kernel which is almost identical, but can perform `Scan Repeats` repititions of the algorithm **inside of the kernel**. However, as this can sometimes mean using an additional register to control the loop or additional computation to limit indexes to the buffer, **this is only an approximation of the kernel**. See Testing Methodology.

</details>

<!-- This content will not appear in the rendered Markdown -->
<!-- This content will not appear in the rendered Markdown -->
<!-- This content will not appear in the rendered Markdown -->
<!-- This content will not appear in the rendered Markdown -->
<!-- This content will not appear in the rendered Markdown -->
<!-- This content will not appear in the rendered Markdown -->
<!-- This content will not appear in the rendered Markdown -->
<!-- This content will not appear in the rendered Markdown -->
# Prefix Sum Survey

A prefix sum blah blah blah. Exclusive sum blah blah. Inclusive sum blah blah.

# Basic Scans
<details open>
<summary>

### Kogge-Stone
  
</summary>

![KoggesStoneImage](https://user-images.githubusercontent.com/68340554/224911618-6f54231c-251f-4321-93ec-b244a0af49f7.png)
  
```HLSL
[numthreads(GROUP_SIZE, 1, 1)]
void KoggeStone(int3 gtid : SV_GroupThreadID)
{
    for (int j = 1; j < e_size; j <<= 1)
    {
        if (gtid.x + j < e_size)
            prefixSumBuffer[gtid.x + j] += prefixSumBuffer[gtid.x];
        DeviceMemoryBarrierWithGroupSync();
    }
}
```
  
</details>

<details>

<summary>

### Sklansky

</summary>


![SklanskyFinal](https://user-images.githubusercontent.com/68340554/224912079-b1580955-b702-45f9-887a-7c1003825bf9.png)

```HLSL
[numthreads(GROUP_SIZE, 1, 1)]
void Sklansky(int3 gtid : SV_GroupThreadID)
{
    int offset = 0;
    for (int j = 1; j < e_size; j <<= 1)
    {
        if ((gtid.x & j) != 0 && gtid.x < e_size)
            prefixSumBuffer[gtid.x] += prefixSumBuffer[((gtid.x >> offset) << offset) - 1];
        DeviceMemoryBarrierWithGroupSync();
        ++offset;
    }
}
```

</details>

<details>

<summary>

### Brent-Kung

</summary>

![BrentKungImage](https://user-images.githubusercontent.com/68340554/224912128-73301be2-0bba-4146-8e20-2f1f3bc7c549.png)

```HLSL
//the classic
[numthreads(GROUP_SIZE, 1, 1)]
void BrentKungBlelloch(int3 gtid : SV_GroupThreadID)
{
    //Upsweep
    if (gtid.x < (e_size >> 1))
        prefixSumBuffer[(gtid.x << 1) + 1] += prefixSumBuffer[gtid.x << 1];
    
    int offset = 1;
    for (int j = e_size >> 2; j > 0; j >>= 1)
    {
        DeviceMemoryBarrierWithGroupSync();
        if (gtid.x < j)
            prefixSumBuffer[(((gtid.x << 1) + 2) << offset) - 1] += prefixSumBuffer[(((gtid.x << 1) + 1) << offset) - 1];
        ++offset;
    }
    //Downsweep
    for (j = 1; j < e_size; j <<= 1)
    {
        --offset;
        DeviceMemoryBarrierWithGroupSync();
        if (gtid.x < j)
            prefixSumBuffer[(((gtid.x << 1) + 3) << offset) - 1] += prefixSumBuffer[(((gtid.x << 1) + 2) << offset) - 1];
    }
}
```

</details>

<details>

<summary>

### Reduce Scan

</summary>

![ReduceScanFinal](https://user-images.githubusercontent.com/68340554/224912530-2e1f2851-f531-4271-8246-d13983ccb584.png)

```HLSL
[numthreads(GROUP_SIZE, 1, 1)]
void ReduceScan(int3 gtid : SV_GroupThreadID)
{
    //cant be less than 2
    int spillFactor = 3;
    int spillSize = e_size >> spillFactor;
    
    //Upsweep until desired threshold
    if (gtid.x < (e_size >> 1))
        prefixSumBuffer[(gtid.x << 1) + 1] += prefixSumBuffer[(gtid.x << 1)];
    AllMemoryBarrierWithGroupSync();
    
    int offset = 1;
    for (int j = e_size >> 2; j > spillSize; j >>= 1)
    {
        if (gtid.x < j)
            prefixSumBuffer[(((gtid.x << 1) + 2) << offset) - 1] += prefixSumBuffer[(((gtid.x << 1) + 1) << offset) - 1];
        AllMemoryBarrierWithGroupSync();
        ++offset;
    }
    
    //Pass intermediates into secondary buffer
    if (gtid.x < j)
    {
        const int t = (((gtid.x << 1) + 2) << offset) - 1;
        g_reduceValues[gtid.x] = prefixSumBuffer[t] + prefixSumBuffer[(((gtid.x << 1) + 1) << offset) - 1];
        prefixSumBuffer[t] = g_reduceValues[gtid.x];
    }
    AllMemoryBarrierWithGroupSync();
    
    //Reduce intermediates
    offset = 0;
    for (j = 1; j < spillSize; j <<= 1)
    {
        if ((gtid.x & j) != 0 && gtid.x < spillSize)
            g_reduceValues[gtid.x] += g_reduceValues[((gtid.x >> offset) << offset) - 1];
        AllMemoryBarrierWithGroupSync();
        ++offset;
    }
    
    //Pass in intermediates and downsweep
    offset = spillFactor - 2;
    const int t = (((gtid.x << 1) + 2) << offset) + (1 << offset + 1) - 1;
    if (t  < e_size)
        InterlockedAdd(prefixSumBuffer[t], g_reduceValues[(t >> spillFactor) - 1]);
    
    for (j = spillSize << 1; j < e_size; j <<= 1)
    {
        AllMemoryBarrierWithGroupSync();
        if (gtid.x < j)
            prefixSumBuffer[(((gtid.x << 1) + 3) << offset) - 1] += prefixSumBuffer[(((gtid.x << 1) + 2) << offset) - 1];
        offset--;
    }
}
```

</details>

<details>

<summary>

### Raking Reduce-Scan

</summary>

![RakingReduceScan](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/3bc46762-1d61-41aa-aee5-1c492ff76ad6)

```HLSL
[numthreads(LANE_COUNT, 1, 1)]
void RakingReduce(int3 gtid : SV_GroupThreadID)
{
    const int partitionSize = e_size >> LANE_LOG;
    const int partStart = partitionSize * gtid.x;
    const int partEnd = (gtid.x + 1) * partitionSize - 1;
    
    //Per-thread serial reductions
    for (int j = partStart + 1; j <= partEnd; ++j)
        prefixSumBuffer[j] += prefixSumBuffer[j - 1];
    
    //Single Kogge-Stone on the aggregates
    prefixSumBuffer[partEnd] += WavePrefixSum(prefixSumBuffer[partEnd]);
    
    //Per-thread serial propogation
    if (gtid.x > 0)
        for (j = partStart; j < partEnd; ++j)
            prefixSumBuffer[j] += prefixSumBuffer[partStart - 1];
}
```

</details>


<details>

<summary>

### TItle

</summary>

code + image

</details>

# Warp-Synchronized Scans
  
<details>

<summary>

### Warp-Sized-Radix Brent-Kung

</summary>

![RadixBrentKung](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/d246240a-c088-4a74-a4c9-3cc2de12d1c9)

```HLSL
[numthreads(GROUP_SIZE, 1, 1)]
void RadixBrentKungLarge(int3 gtid : SV_GroupThreadID)
{
    //Upsweep
    //Warp-sized-radix KoggeStone embedded into BrentKung
    int offset = 0;
    for (int j = e_size; j > 1; j >>= LANE_LOG)
    {
        for (int i = gtid.x; i < j; i += GROUP_SIZE)
        {
            const int t = ((i + 1) << offset) - 1;
            prefixSumBuffer[t] += WavePrefixSum(prefixSumBuffer[t]);
        }
        DeviceMemoryBarrierWithGroupSync();
        offset += LANE_LOG;
    }
    
    //Downsweep
    //Warp-sized-radix propogation fans
    offset = LANE_LOG;
    for (j = 1 << LANE_LOG; j < e_size; j <<= LANE_LOG)
    {
        for (int i = gtid.x + j; i < e_size; i += GROUP_SIZE)
            if ((i & (j << LANE_LOG) - 1) >= j)         
                if ((i + 1 & j - 1) != 0)                
                    prefixSumBuffer[i] += prefixSumBuffer[((i >> offset) << offset) - 1];
        DeviceMemoryBarrierWithGroupSync();
        offset += LANE_LOG;
    }
}
```

</details>
  
<details>

<summary>

### Warp-Sized-Radix Brent-Kung with Fused Upsweep-Downsweep

</summary>

![RadixBKFused](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/4ccfaf59-9864-4405-a89f-f950c70cda2b)

```HLSL
[numthreads(GROUP_SIZE, 1, 1)]
void RadixBrentKungFused(int3 gtid : SV_GroupThreadID)
{
    int offset = 0;
    for (int j = 1; j < (e_size >> 1); j <<= LANE_LOG)
    {
        for (int i = gtid.x; i < (e_size >> offset); i += GROUP_SIZE)
            prefixSumBuffer[((i + 1) << offset) - 1] += WavePrefixSum(prefixSumBuffer[((i + 1) << offset) - 1]);
        DeviceMemoryBarrierWithGroupSync();
        
        for (int i = gtid.x + j; i < e_size; i += GROUP_SIZE)
            if ((i & (j << LANE_LOG) - 1) >= j)         
                if ((i + 1 & j - 1) != 0)                
                    prefixSumBuffer[i] += WaveReadLaneFirst(prefixSumBuffer[((i >> offset) << offset) - 1]);
        offset += LANE_LOG;
    }
    DeviceMemoryBarrierWithGroupSync();
    
    for (int i = gtid.x + j; i < e_size; i += GROUP_SIZE)              
        prefixSumBuffer[i] += WaveReadLaneFirst(prefixSumBuffer[((i >> offset) << offset) - 1]);
}
```

</details>
  
<details>

<summary>

### Warp-Sized-Radix Sklansky

</summary>

![WarpSklansky](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/e7dd3c56-334f-431a-a7bf-0d30fa64ea9a)

```HLSL
[numthreads(GROUP_SIZE, 1, 1)]
void RadixSklanskyAdvanced(int3 gtid : SV_GroupThreadID)
{
    //Warp-sized radix Kogge-Stone
    for (int i = gtid.x; i < e_size; i += GROUP_SIZE)
        prefixSumBuffer[i] += WavePrefixSum(prefixSumBuffer[i]);
    DeviceMemoryBarrierWithGroupSync();
    
    int offset = LANE_LOG;
    for (int j = 1 << LANE_LOG; j < e_size; j <<= 1)
    {
        for (int i = gtid.x; i < (e_size >> 1); i += GROUP_SIZE)
        {
            const int t = ((((i >> offset) << 1) + 1) << offset) + (i & (1 << offset) - 1);
            prefixSumBuffer[t] += WaveReadLaneFirst(prefixSumBuffer[((t >> offset) << offset) - 1]);
        }
        DeviceMemoryBarrierWithGroupSync();
        ++offset;
    }
}
```

</details>
  
<details>

<summary>

### Warp-Sized-Radix Raking Reduce-Scan

</summary>

![WarpRakingReduce](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/72997c9e-ae94-41f0-83e8-c1122530f2e4)
```HLSL
[numthreads(LANE_COUNT, 1, 1)]
void RadixRakingReduce(int3 gtid : SV_GroupThreadID)
{
    const int partitions = e_size >> LANE_LOG;
    
    //Single kogge-stone warp scan without passing in passing in aggregate
    prefixSumBuffer[gtid.x] += WavePrefixSum(prefixSumBuffer[gtid.x]);
    
    //Walk up partitions, passing in the agrregate as we go
    for (int partitionIndex = 1; partitionIndex < partitions; ++partitionIndex)
    {
        const int partitionStart = partitionIndex << LANE_LOG;
        const int t = gtid.x + partitionStart;
        prefixSumBuffer[t] += WavePrefixSum(prefixSumBuffer[t]) + prefixSumBuffer[partitionStart - 1];
    }
}
```

</details>

# Block-Level Scan Pattern

<details>

<summary>

### Block-Level Scan Pattern

</summary>

![Block Level 1](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/306d0908-da29-45a2-9f79-fea4c3856560)

Explanation goes here...

![Block Level 2](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/d5d47250-f5ff-4156-85cd-6ba2639fa237)
```HLSL
#define PARTITION_SIZE          32  //The number of elements in a partition
#define GROUP_SIZE              16  //The number of threads in a threadblock
#define LANE_COUNT              4   //The number of threads in a warp
#define LANE_MASK               3   //A bit-mask used to determine a thread's lane
#define LANE_LOG                2   //log2(LANE_COUNT)
#define WAVES_PER_GROUP         4   //The number of warps per threadblock
#define WAVE_PARTITION_SIZE     8   //The number of elements a warp processes per partition
#define WAVE_PART_LOG           3   //log2(WAVE_PARTITION_SIZE)

#define LANE                (gtid.x & LANE_MASK)
#define WAVE_INDEX          (gtid.x >> LANE_LOG)
#define SPINE_INDEX         (((gtid.x + 1) << WAVE_PART_LOG) - 1)
#define PARTITIONS          (e_size >> PART_LOG)
#define PARTITION_START     (partitionIndex << PART_LOG)
#define WAVE_PART_START     (WAVE_INDEX << WAVE_PART_LOG)
#define WAVE_PART_END       (WAVE_INDEX + 1 << WAVE_PART_LOG)

extern int e_size;                                //The size of the buffer, this is set by host CPU code
RWBuffer<uint> b_prefixSum;                       //The buffer to be prefix_summed
groupshared uint g_sharedMem[PARTITION_SIZE];     //The array of shared memory we use to compute intermediate values
  
[numthreads(GROUP_SIZE, 1, 1)]
void BlockWarpRakingReduce(int3 gtid : SV_GroupThreadID)
{
    uint aggregate = 0;
    for (int partitionIndex = 0; partitionIndex < PARTITIONS; ++partitionIndex)
    {
        g_sharedMem[LANE + WAVE_PART_START] = b_prefixSum[LANE + WAVE_PART_START + PARTITION_START];
        g_sharedMem[LANE + WAVE_PART_START] += WavePrefixSum(g_sharedMem[LANE + WAVE_PART_START]);
        
        for (int i = LANE + WAVE_PART_START + LANE_COUNT; i < WAVE_PART_END; i += LANE_COUNT)
        {
            g_sharedMem[i] = b_prefixSum[i + PARTITION_START];
            g_sharedMem[i] += WavePrefixSum(g_sharedMem[i]) + WaveReadLaneFirst(g_sharedMem[i - 1]);
        }
        GroupMemoryBarrierWithGroupSync();
        
        if (gtid.x < WAVES_PER_GROUP)
            g_sharedMem[SPINE_INDEX] += WavePrefixSum(g_sharedMem[SPINE_INDEX]) + aggregate;
        GroupMemoryBarrierWithGroupSync();
        
        const uint prev = WAVE_INDEX ? WaveReadLaneFirst(g_sharedMem[WAVE_PART_START - 1]) : aggregate;
        for (int i = LANE + WAVE_PART_START; i < WAVE_PART_END; i += LANE_COUNT)
            b_prefixSum[i + PARTITION_START] = g_sharedMem[i] + (i < WAVE_PART_END - 1 ? prev : 0);
        
        aggregate = WaveReadLaneFirst(g_sharedMem[PARTITION_SIZE - 1]);
        GroupMemoryBarrierWithGroupSync();
    }
}
```

</details>
  
# Device-Level Scan Pattern
  
<details>

<summary>

### Device-Level Scan Pattern

</summary>

## Reduce
  
![Device 1](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/a95e4647-440b-42d9-ae3e-b7a34247141d)

```HLSL
#define SUB_PARTITION_SIZE      32  //The size of the threadblock's subpartition.
#define GROUP_SIZE              16  //The number of threads in a threadblock
#define THREAD_BLOCKS           8   //The number of threadblocks dispatched
#define TBLOCK_LOG              3   //log2(THREAD_BLOCKS)

#define LANE_COUNT              4   //The number of threads in a warp
#define LANE_MASK               3   //A bit-mask used to determine a thread's lane
#define LANE_LOG                2   //log2(LANE_COUNT)
#define WAVES_PER_GROUP         4   //The number of warps per threadblock
#define WAVE_PARTITION_SIZE     8   //The number of elements a warp processes per subpartition
#define WAVE_PART_LOG           3   //log2(WAVE_PARTITION_SIZE)

#define PARTITION_SIZE      (e_size >> TBLOCK_LOG)
#define LANE                (gtid.x & LANE_MASK)
#define WAVE_INDEX          (gtid.x >> LANE_LOG)
#define SPINE_INDEX         (((gtid.x + 1) << WAVE_PART_LOG) - 1)
#define WAVE_PART_START     (WAVE_INDEX << WAVE_PART_LOG)
#define WAVE_PART_END       (WAVE_INDEX + 1 << WAVE_PART_LOG)
#define PARTITION_START     (gid.x * PARTITION_SIZE)
#define SUB_PART_START      (subPartitionIndex << SUB_PART_LOG)
#define SUB_PARTITIONS      (PARTITION_SIZE >> SUB_PART_LOG)

extern int e_size;                                    //The size of the buffer, this is set by host CPU code
RWBuffer<uint> b_prefixSum;                           //The buffer to be prefix_summed
groupshared uint g_sharedMem[SUB_PARTITION_SIZE];     //The array of shared memory we use to compute intermediate values
  
[numthreads(GROUP_SIZE, 1, 1)]
void DeviceMainReduce(int3 gtid : SV_GroupThreadID, int3 gid : SV_GroupID)
{
    uint waveAggregate = 0;
    const int partitionEnd = (gid.x + 1) * PARTITION_SIZE;
    for (int j = gtid.x + PARTITION_START; j < partitionEnd; j += GROUP_SIZE)
        waveAggregate += WaveActiveSum(b_prefixSum[j]);
    GroupMemoryBarrierWithGroupSync();
    
    if (LANE == 0)
        g_sharedMem[WAVE_INDEX] = waveAggregate;
    GroupMemoryBarrierWithGroupSync();
    
    if (gtid.x < WAVES_PER_GROUP)
        g_sharedMem[gtid.x] = WaveActiveSum(g_sharedMem[gtid.x]);
```
  
Explanation goes here...
  
![Device 2](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/0641660d-db0d-48ad-8ecf-3d6312825ffd)

```HLSL
    if (gtid.x == 0)
    {
        InterlockedAdd(b_state[gid.x], g_sharedMem[gtid.x]);
        InterlockedAdd(b_state[THREAD_BLOCKS], 1, g_sharedMem[0]);
    }
    GroupMemoryBarrierWithGroupSync();
    
    if (WaveReadLaneFirst(g_sharedMem[0]) == THREAD_BLOCKS - 1)
    {
        GroupMemoryBarrierWithGroupSync();
        if (gtid.x < THREAD_BLOCKS)
        {
            g_sharedMem[gtid.x] = b_state[gtid.x];
            g_sharedMem[gtid.x] += WavePrefixSum(g_sharedMem[gtid.x]);
        }
        GroupMemoryBarrierWithGroupSync();
        
        if (gtid.x < (THREAD_BLOCKS >> LANE_LOG))
            g_sharedMem[((gtid.x + 1) << LANE_LOG) - 1] += WavePrefixSum(g_sharedMem[((gtid.x + 1) << LANE_LOG) - 1]);
        GroupMemoryBarrierWithGroupSync();
        
        if (gtid.x < THREAD_BLOCKS)
            b_state[gtid.x] = g_sharedMem[gtid.x] +
                (LANE < LANE_MASK && gtid.x > LANE_MASK ? WaveReadLaneFirst(g_sharedMem[gtid.x - 1]) : 0);
    }
}
```
  
Explanation
  
## Scan
  
![Device 3](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/e80f6ece-b2db-48c4-98e6-be20a6b19fb8)

```HLSL
[numthreads(GROUP_SIZE, 1, 1)]
void DeviceMainScan(int3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint aggregate = gid.x ? b_state[gid.x - 1] : 0;
    for (int subPartitionIndex = 0; subPartitionIndex < SUB_PARTITIONS; ++subPartitionIndex)
        {
            g_sharedMem[LANE + WAVE_PART_START] = b_prefixSum[LANE + WAVE_PART_START + SUB_PART_START + PARTITION_START];
            g_sharedMem[LANE + WAVE_PART_START] += WavePrefixSum(g_sharedMem[LANE + WAVE_PART_START]);
          
            for (int i = LANE + WAVE_PART_START + LANE_COUNT; i < WAVE_PART_END; i += LANE_COUNT)
            {
                g_sharedMem[i] = b_prefixSum[i + SUB_PART_START + PARTITION_START];
                g_sharedMem[i] += WavePrefixSum(g_sharedMem[i]) + WaveReadLaneFirst(g_sharedMem[i - 1]);
            }
            GroupMemoryBarrierWithGroupSync();
        
            if (gtid.x < WAVES_PER_GROUP)
            g_sharedMem[SPINE_INDEX] += WavePrefixSum(g_sharedMem[SPINE_INDEX]) + aggregate;
            GroupMemoryBarrierWithGroupSync();
        
            const uint prev = WAVE_INDEX ? WaveReadLaneFirst(g_sharedMem[WAVE_PART_START - 1]) : aggregate;
            for (int i = LANE + WAVE_PART_START; i < WAVE_PART_END; i += LANE_COUNT)
              b_prefixSum[i + SUB_PART_START + PARTITION_START] = g_sharedMem[i] + (i < WAVE_PART_END - 1 ? prev : 0);
        
            aggregate = WaveReadLaneFirst(g_sharedMem[SUB_PARTITION_SIZE - 1]);
            GroupMemoryBarrierWithGroupSync();
        }
}
```
  
</details>
  
# Chained Scan With Decoupled Lookback
  
<details>

<summary>

### Chained Scan With Decoupled Lookback

</summary>

## Acquiring Partition Index
  
![Chained 1](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/02aa0608-247b-4b31-81c0-5fc7c3a696a9)
  
```HLSL

```
  
Explanation goes here...
  
## Reduce
  
![Chained 2](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/68788f9b-01cb-4796-ad9b-e0fb515f67ec)

Explanation goes here...
  
```HLSL

```
 
## LookBack
![Aggregates](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/6836fa80-e324-413e-a882-0c8b9f6e4db0)
  
Explanation goes here...


![Lookback](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/bd935510-c851-459b-9c66-58963696b4fd)

  
Explanation goes here...
  
## Propagation
  
![Propagation](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/c495cdc8-7dfd-461a-87ff-23bff2158e4b)
  
Explanation goes here...
  
```HLSL

```

</details>

# Testing Methodology
