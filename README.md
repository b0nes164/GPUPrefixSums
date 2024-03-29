# GPU Prefix Sums
![Prefix Sum Speed Comparison](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/f2088c9d-a3e5-4b6c-b318-4765501f9fb6)

This project is a survey of GPU prefix sums, ranging from the warp to the device level, with the aim of providing developers an uncompiled look at modern prefix sum implementations. In particular, this project was inspired by Duane Merill's [research](https://libraopen.lib.virginia.edu/downloads/6t053g00z) and includes implementations of Merill and Garland's [Chained Scan with Decoupled Lookback](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back), which is how we are able to reach speeds approaching `MemCopy()`. Finally, this project was written in HLSL for compute shaders, though with reasonable knowledge of GPU programming it is easily portable. 

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
  
  <summary><s>Currently, this project does not work on AMD or integrated graphics hardware.</s> Support for AMD and Nvidia hardware added! <b>ALGORITHMS UNTESTED ON NON-NVIDIA HARDWARE. TEST BEFORE USING!</b></summary>
  
</br> All algorithms above the warp level or not in the survey *should* support any wave size >= 32. The survey or the warp level prefix sums will still need to be changed manually because some kernels dispatch exactly wave width threads. Due to difficulties activating a Unity license on my AMD computer, I have not been able to test the algorithms on that computer.
 
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
  
  <summary><s>All scans are inclusive.</s> Exclusive prefix sums added!</summary>

  </br>~~I have exclusive versions of the scans, but I would like to polish them a little more and will release them sometime in the future.~~
  
</details>

<details>
  
  <summary>Headless compute versions for Vulkan and DX12 coming eventually!</summary>

  </br>I plan to add these... sometime?
  
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

# To Use This Project in Unity

1. Download or clone the repository.
2. Drag the contents of `Unity` into a desired folder within a Unity project.
3. The survey of underlying prefix sum variants can be found in the `Survey` folder. All other prefix sums are organized into folders based on their device utilization level: warp/wave, block, and device.
4. Survey or warp/wave level prefix sums must be manually changed depending on the hardware. (This is because we dispatch exactly wave width threads).
5. Every scan variant has a compute shader and a dispatcher. Attach the desired scan's dispatcher to an empty game object. All scan dispatchers are named  `ScanNameHere.cs`.
6. Attach the matching compute shader to the game object. All compute shaders are named `ScanNameHere.compute`. The dispatcher will return an error if you attach the wrong shader.
7. Ensure the slider is set to a nonzero value.

If you did this correctly you should see this in the inspector:

![Example1](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/acd77d76-5604-409f-87be-61c72a86f20f)

<details>

<summary>

## Testing Suite

</summary>
  
![Tests](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/067b9d12-582d-4c30-9529-519398ef0d0b)


Every scan dispatcher has a suite of tests which can be controlled directly from the inspector. To reduce lag on large input sizes, validation tests will only print the first 1024 errors found.

+ `Validate Prefix Sum` performs a prefix sum at the current `Input Size`, then validates the output.

+ `Validate Prefix Sum Random` performs a prefix sum on a buffer of random values at the current `Input Size`. By default, the buffer is filled with the value 1. This makes error checking very simple, because the correct prefix sum is the sequence of positive integers. However this makes some other errors possible, so to cover our bases we include this test.

+ `Debug Prefix Sum` performs a prefix sum at the current `Input Size`, then prints the output into the debug log without any validation.

+ `Debug State` is exclusive to _ChainedScanWithDecoupledLookback_, and prints the threadblock aggregate values in the auxillary buffer. It should be used to verify that the lookback phase is executing correctly.
  
+ `Timing Test` times the execution of the prefix sum. However, this is not the true execution time of the algorithm, see testing methodology section for more information.

+ `Validate Powers of Two` performs a prefix sum at an input size of each power of two from 21 to 28. Useful as a quick measure of correct execution of the prefix sum.

+ `Validate All Off Sizes` performs a series of tests to ensure that the perfix sum correctly handles non-powers-of-two buffer sizes.

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
![Prefix](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/a2cfdec0-53ed-4048-bc5e-8efda30f7cf4)

A prefix sum, also called a scan, is a running total of a sequence of numbers at the n-th element. If the prefix sum is *inclusive* the n-th element is included in that total, if it is *exclusive,* the n-th element is not included. The prefix sum is one of the most important alogrithmic primitives in parallel computing, underpinning everything from [sorting](https://arxiv.org/ftp/arxiv/papers/2206/2206.01784.pdf), to [compression](https://arxiv.org/abs/1311.2540), to [graph traversal](https://research.nvidia.com/sites/default/files/pubs/2012-02_Scalable-GPU-Graph/ppo213s-merrill.pdf). 

In this survey, we will build from simple scans then work our way up from the warp to the device level, eventually ending with the current state-of-the art algorithm, Merill and Garland's **[Chained Scan with Decoupled Lookback](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back)**. 

**Note: The author would like to apologize in advance for interchanging the terms prefix-sum and scan, warp and wave, and threadblocks and threadgroups. Warp and wave are Nvidia and Microsoft's (respective) terminology for sets of threads executed in parallel on the processor. Similarly threadblocks and threadgroups are Nvidia and Microsoft's (respective) terminology for sets of warps. A prefix sum is a scan whose binary associative operator is addition.** 
# Basic Scans

We begin with "basic scans," scans which are parallelized, but agnostic of GPU-specific implementation features. While these exact scans will not appear in our final implementation, the underlying patterns still form the basis of the more advanced scans.

In this section, we introduce the diagram pattern that is throughout the rest of the survey. Every diagram consists of an input buffer with all elements initialized to 1, with arrows representing additions made by threads. We initialize the buffer this way because it allows us to easily verify the correctness of the sum: the correct inclusive sum of the n-th element is always n + 1. Every diagram will specify the number of threadblocks, and the number of threads per threadblock (block size/group size). The size of the scan will always be represented by the variable `e_size`, and the buffer on which we are performing the scan will always be named `prefixSumBuffer` or `b_prefixSum`.

As you go through the code examples you will notice the functions `DeviceMemoryBarrierWithGroupSync()`, `AllMemoryBarrierWithGroupSync()`, or `GroupMemoryBarrierWithGroupSync()`. These are intrinsic [barrier functions](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/allmemorybarrierwithgroupsync) provided by HLSL that allow us to control the flow of execution of threads within the same threadblock. A common misconception is that the `Device` or `All` included in the function name refers to the level of synchronization across the GPU: that `All` must synchronize *all* threads across *all* blocks; that `Device` must synchronize all threads across the `device`. **This is not true.** These functions operate only on the threadblock level: **they only synchronize threads within the same threadblock.** What these functions actually specify is the level of synchronization on **memory accesses**, synchronizing **device** memory read/writes, **groupshared** memory read/writes, or **all** memory read/writes.

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

# Warp-Synchronized Scans
  
When a kernel is dispatched, an SM executes multiple, independent copies of the program called threads in parallel groups of 32 (varies by hardware) called warps. GPU’s follow a hybridization of the single instruction, multiple data (SIMD) and single program, multiple thread (SPMD) models in that the SM will attempt to execute all the threads within the same warp in lockstep (SIMD), but will allow the threads to diverge and take different instruction paths if necessary (SPMD). However, divergence is generally undesirable because the SM cannot run both branches in parallel, but instead disables the threads of the opposing branch in order to run the threads of the current branch in lockstep, effectively running the branches serially.

What is important for us is that threads within the same warp are inherently synchronized. This means that we do not have to place as many barriers to guaruntee the correct execution of our program. More importantly, threads within the same warp are able to effeciently communicate with each other   using hardware intrinsic functions. These functions allow us to effeciently broadcast one value to all other threads, `WaveReadLaneFirst()`, and can even perform a warp wide exclusive prefix sum `WavePrefixSum()`. To incorporate these techniques into our algorithm, we change the radix of the network from base 2 to match the size of the warp.
  
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

### Warp-Sized-Radix Serial

</summary>

![RadixSerial](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/11dccbcb-7ab4-4a99-aacb-704d0c4581fd)

```HLSL
[numthreads(LANE_COUNT, 1, 1)]
void RadixSerial(int3 gtid : SV_GroupThreadID)
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
  
<details>

<summary>

### Warp-Sized-Radix Raking Reduce-Scan

</summary>

![WarpRakingReduce](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/72997c9e-ae94-41f0-83e8-c1122530f2e4)
```HLSL
#define LANE                (gtid.x & LANE_MASK)
#define WAVE_INDEX          (gtid.x >> LANE_LOG)
#define WAVE_PART_START     (WAVE_INDEX << WAVE_PART_LOG)
#define WAVE_PART_END       (WAVE_INDEX + 1 << WAVE_PART_LOG)
#define SPINE_INDEX         (((gtid.x + 1) << WAVE_PART_LOG) - 1)

[numthreads(GROUP_SIZE, 1, 1)]
void RadixRakingReduce(int3 gtid : SV_GroupThreadID)
{
    g_sharedMem[LANE + WAVE_PART_START] = b_prefixSum[LANE + WAVE_PART_START];
    g_sharedMem[LANE + WAVE_PART_START] += WavePrefixSum(g_sharedMem[LANE + WAVE_PART_START]);

    for (int i = LANE + WAVE_PART_START + LANE_COUNT; i < WAVE_PART_END; i += LANE_COUNT)
    {
        g_sharedMem[i] = b_prefixSum[i];
        g_sharedMem[i] += WavePrefixSum(g_sharedMem[i]) + WaveReadLaneFirst(g_sharedMem[i - 1]);
    }
    GroupMemoryBarrierWithGroupSync();

    if (gtid.x < WAVES_PER_GROUP)
        g_sharedMem[SPINE_INDEX] += WavePrefixSum(g_sharedMem[SPINE_INDEX]) + aggregate;
    GroupMemoryBarrierWithGroupSync();

    const uint prev = WAVE_INDEX ? WaveReadLaneFirst(g_sharedMem[WAVE_PART_START - 1]) : aggregate;
    for (int i = LANE + WAVE_PART_START; i < WAVE_PART_END; i += LANE_COUNT)
        b_prefixSum[i] = g_sharedMem[i] + (i < WAVE_PART_END - 1 ? prev : 0);
}
```

</details>

# Block-Level Scan Pattern

![MemoryHierarchyJiaEtAl](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/e442873a-5212-40ec-b510-909f0b37582e)

<br>Up until this point, all of the scan patterns shown operate on a single threadblock, however they are not true "block-level scans" because they are agnostic of the last critical component of GPU optimization: the GPU memory hierarchy. As is shown in the figure above from Jia et al's [paper](https://arxiv.org/abs/1903.07486), the GPU has various memory levels, and is shown in the following table, each level has differing memory access latencies. **Note these are ballpark estimates, and performance differs significantly between hardwares.** 


| | Simple Arithmetic Instruction | Registers | L1 Cache |  Shared Memory | L2 Cache | Device Memory|
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |
| Latency in Cycles  | ~4  | ~1-3 | ~30 | ~20 | ~200 | ~600 |


## Hiding GPU Memory Latency
Unlike CPU’s which hide their memory latency through relatively large caches and branch prediction, GPU's employ massive amounts of memory bandwith to saturate SM's with data, and hide their memory latency through a combination of *occupancy* and careful use of registers and shared memory.

### Occupancy:
Each SM can host multiple active warps simultaneously, typically up to 32 or 64, and the number of warps hosted is known as the *occupancy* rate. However a typical SM has only 2 - 4 warp scheduling units, meaning that only 2 -4 of those hosted warps are being executed at any given time. This allows the SM execute "ready" warps while others are waiting to read/write data back to global memory. With sufficient arithmetic pressure and/or occupancy, the entire read/write time can be hidden by the work of other warps. Under ideal circumstances, memory latency is fully hidden by the computational work of the warps, or the memory bandwidth is fully saturated.

### Registers/Shared Memory:
Shared memory is memory that is visible to all threads within a threadblock, and registers are memory that is visible only to a thread. A typical modern Nvidia GPU has 96kb of memory that is divided between shared memory and L1. The specific proportion between the two is called the *carveout*, and typically this is 32kb shared memory, 64 kb L1. Because of the dramatically reduced latency of more local memory, effecient GPU algorithms limit device level memory access as much as possible, typically to the extent that the only times device memory is accessed is to read input data and write output data. 

### Memory Bandwidth Effeciency:
Lastly, memory latency aside, we want to maximize the memory bandwidth efficiency of our reads and writes. To do so, we want our reads and writes to be *coalesced*, which means that each warp reads and writes contiguous sections of memory, rather than each thread accessing discontinous memory locations. Furthermore, we use *vectorized/multiword* reads and writes: instead of reading only 4 bytes or a single 32-bit variable at a time, we read 16 bytes or four 32-bit variables a time. For an algorithm like the prefix sum, which is computationally very light, memory access is the limiting factor on the performance of the algorithm, and thus effecient memory access has a dramatic effect on performance.

## In Practice

### Occupancy:

We want to make sure that each SM hosts up to its maximum number of warps. To do so we carefully limit the number of registers each thread uses to ensure we do not spillover into device memory, and ensure that memory footprint of all warps can fit on the SM. This also means that we use preprocessor macros to calculate intermediate values to save on valuable register space.

### Shared Memory:

Because we always want to be operating on shared memory, we limit the size of our scan to the maximum size of our shared memory. We then partition our buffer into tiles equal to that size, only accessing device memory to load in new partition tiles or read out processed partition tiles. We work our way serially along the partition tiles, passing in the previous aggregate sum as we go.

### Memory Bandwidth Effeciency:
Warp-Synchronized-Scans have inherently coalesced reads and writes, so no modification is necessary here. Vectorization is quite simple to achieve, but for simplicity I have not included it in these examples. Shared memory has a catch however: [*bank conflicts.* ](https://github.com/Kobzol/hardware-effects-gpu/blob/master/bank-conflicts/README.md). Basically shared memory is organized into banks, which can be thought of as groupings of memory indexes. If multiple threads in the same warp attempt to access indexes within the same bank, a "bank conflict" occurs, and each thread must wait for its turn before performing its access, effectively rendering the operation serial. On a typical modern Nvidia GPU the banks are organized into 4 byte strides, with 32 banks, one for each thread of a warp. So for example, if we use our shared memory to store 32-bit/4-byte values like an `int` then attempt to access indexes `0` and `32` from within the same warp, we will incur a bank conflict. There are different ways of combatting bank conflicts, but we simply use an underlying scan pattern which is mostly conflict free: a warp-sized-radix raking reduce-scan.
  
<details>

<summary>

### Block-Level Scan Pattern

</summary>

## Initialization and First Partition
![Block Level 1](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/306d0908-da29-45a2-9f79-fea4c3856560)

<br>We begin our scan by partitioning the buffer into tiles of size equal to the maximum shared memory size, which in our example is 32. Although in our diagram I show the buffer being partitioned into distinct tiles, in practice all this means is determing the number of partitions required to process the entire buffer, and determing the beginning and ending index of each partition.

We begin our first partition, loading from device memory into shared memory. We perform our prefix sum, then pass our results back into device memory, eliding storage of complete results in shared memory by passing the results directly in device memory. Once this is complete, all threads store the aggregate sum of that partition in a register, then proceed to the next partition tile. 
<br>
<br>
## Second Partition and Onwards
![Block Level 2](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/d5d47250-f5ff-4156-85cd-6ba2639fa237)


The second partition proceeds almost identically to the first, except that along the downsweep we pass in the aggregate from the previous partition and add the current aggregate to our register. This pattern repeats until all partition tiles are complete.
<details>
  
<summary>

### Boilerplate Preprocessor Macros

</summary>

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
```

</details>
    
  
```HLSL
extern int e_size;                                //The size of the buffer, this is set by host CPU code
RWBuffer<uint> b_prefixSum;                       //The buffer to be prefix_summed
groupshared uint g_sharedMem[PARTITION_SIZE];     //The array of shared memory we use for intermediate values
  
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
  
Up until this point, all of the scans we have explored operate on a single threadblock. However, a GPU can host dozens of threadblocks, and so to fully utilize our GPU we need an algorithm which can utilize multiple threadblocks instead of just one. However, this is easier said than done. In terms of general device-level implementation issues, there is the issue of inter-threadblock communication. There is no "threadblock shared memory," nor are there (kosher) device-wide fences that we can use to synchronize threadblocks. More specifically to prefix sums, each element is serially dependent on the sum of preceeding elements. On device-level this means that each threadblock is serially dependent on the result of preceeding threadblocks.

To solve these problems, we use the reduce-then-scan technique from the basic scans section. We begin by partitioning the input among the threadblocks. To overcome the serial depedency of the sums, each threadblock computes the reduction of its partiition amd stores the result in an intermediate device level buffer. We perform a prefix sum across these intermediates and store them. Finally, each threadblock performs a prefix sum across its partition tile, and passes in the correct preceeding sum from the intermediate value buffer. Because only the reduced intermediate value is passed between threadblocks, the device-memory access overhead is neglible. In order to main coherence between the threadblocks, we seperate the reduce phase and the scan phase into individual kernels.
  
<details>

<summary>

### Device-Level Scan Pattern

</summary>

## Reduce

![Device 1](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/f2be3fc7-33b5-427a-bbf8-7bfca31d9d55)

<details>

<summary>

### Boilerplate Preprocessor Macros

</summary>

```HLSL
#define SUB_PARTITION_SIZE      32  //The size of the threadblock's subpartition.
#define GROUP_SIZE              16  //The number of threads in a threadblock
#define THREAD_BLOCKS           4   //The number of threadblocks dispatched
#define SUB_PART_LOG            5   //log2(SUB_PARTITION_SIZE)
#define TBLOCK_LOG              2   //log2(THREAD_BLOCKS)

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
```

</details>
  
```HLSL
extern int e_size;                                    //The size of the buffer, this is set by host CPU code
RWBuffer<uint> b_prefixSum;                           //The buffer to be prefix_summed
globallycoherent RWBuffer<uint> b_state;              //The buffer used for threadblock aggregates.
groupshared uint g_sharedMem[SUB_PARTITION_SIZE];     //The array of shared memory we use for intermediate values
  
[numthreads(GROUP_SIZE, 1, 1)]
void DeviceSimpleReduce(int3 gtid : SV_GroupThreadID, int3 gid : SV_GroupID)
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
  
Each threadblock/threadgroup, identified by its group id `gid`, calculates the reduction of its partition. To do this, we use the `WaveActiveSum` intrinsic function to sum the elements within a warp. When all warps finish processing the partition, the first thread in each warp places its aggregate sum into shared memory, then the first warp in a threadblock sums the warp aggregates and places the final value into the intermediate device buffer.
#

![Device 2](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/beefc3d8-a474-429c-8cf9-c101612e607f)

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
  
After a threadblock inserts its partition aggregate into the intermediate buffer, it atomically bumps an index value in device memory. By doing this, we can keep track of how many threadblocks have completed their reductions. The last threadblock to complete its reduction is then given the task of performing the prefix sum of the intermediate values. Because the size of the intermediate buffer is so small, we do not need to use the entire GPU to compute it.
<br>
<br>
## Scan
  
![Device3](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/b8c82585-1773-401b-8049-c096682ce036)

```HLSL
[numthreads(GROUP_SIZE, 1, 1)]
void DeviceSimpleScan(int3 gtid : SV_GroupThreadID, int3 gid : SV_GroupID)
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

The diagram above shows the state of the third threadblock partway through its partition. The scan kernel is an exact copy of the block-level scan except that we pass in the aggregate value from the intermediate buffer and we only process the threadblock's partition rather than the whole input. 
<br>
</details>
  
# Chained Scan With Decoupled Lookback

Finally we reach *Chained Scan with Decoupled LookBack.* (CSDL) To understand why CSDL is an improvement over reduce-then-scan (RS) let us reexamine the RS algorithm paying close attention the device memory accesses. Looking at RS we observe that we must read *n* elements during the reduce phase, then read and write *n* elements again during scan, for a total of *3n* total device level data movement. CSDL makes the observation that the *n* reads during the reduce phase are wasted, and that with careful ordering of threadblocks, it is possible to propogate aggregate sums *while* we are performing the scan. By doing this, we reduce the overall device-level data movement to *2n*. Because prefix sums are compuationally very light and operate at the limit of the memory bandwidth, reducing the data movement by *n* improves the performance by ~50%. 

The key issue that CSDL solves is the coordination between threadblocks. Instead of dividing the input into equal partitions among the threadblocks, we divide the input into smaller fixed size partition tiles, with the size equal to about the maximum shared memory. Each threadblock is assigned a partition tile. After reading its input into shared memory and computing the aggregate sum, the threadblock posts the value into an intermediate buffer along with a flag value that signals to other threadblocks that the partition tile's aggregate has been posted. Then it *looks back* through the intermediate buffer, summing the preceeding aggregates to determine the correct preceeding sum. Once the preceeding sum has been found, the value is added to the intermediate buffer, and the flag is updated to signal that the inclusive sum is ready and that other *look backs* can stop upon reaching this value. During this time, the partition tile inputs are still resident in shared memory, and once the *look back phase* is complete, the preceeding sum can be used to output the correct prefix sum value.

<details>

<summary>

### Chained Scan With Decoupled Lookback

</summary>

## Acquiring Partition Index
  
![Chained 1](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/f8bda243-7d52-40bb-af52-668ef0c49404)

<details>

<summary>

### Boilerplate Preprocessor Macros

</summary>

```HLSL
#define PARTITION_SIZE          32  //The size of the threadblock's subpartition.
#define GROUP_SIZE              16  //The number of threads in a threadblock
#define THREAD_BLOCKS           4   //The number of threadblocks dispatched
#defien PART_LOG                5   //log2(PARTITION_SIZE)

#define LANE_COUNT              4   //The number of threads in a warp
#define LANE_MASK               3   //A bit-mask used to determine a thread's lane
#define LANE_LOG                2   //log2(LANE_COUNT)
#define WAVES_PER_GROUP         4   //The number of warps per threadblock
#define WAVE_PARTITION_SIZE     8   //The number of elements a warp processes per subpartition
#define WAVE_PART_LOG           3   //log2(WAVE_PARTITION_SIZE)

#define FLAG_NOT_READY  0
#define FLAG_AGGREGATE  1
#define FLAG_INCLUSIVE  2
#define FLAG_MASK       3

#define LANE                (gtid.x & LANE_MASK)
#define WAVE_INDEX          (gtid.x >> LANE_LOG)
#define SPINE_INDEX         (((gtid.x + 1) << WAVE_PART_LOG) - 1)
#define PARTITION_START     (partitionIndex << PART_LOG)
#define WAVE_PART_START     (WAVE_INDEX << WAVE_PART_LOG)
#define WAVE_PART_END       (WAVE_INDEX + 1 << WAVE_PART_LOG)
#define PARTITIONS          (e_size >> PART_LOG)
```

</details>
  
```HLSL
extern int e_size;                              //The size of the buffer, this is set by host CPU code
RWBuffer<uint> b_prefixSum;                     //The buffer to be prefix_summed
globallycoherent RWBuffer<uint> b_state;        //The buffer used for threadblock aggregates.
groupshared uint g_sharedMem[PARTITION_SIZE];   //The array of shared memory we use for intermediate values
groupshared uint g_aggregate;                   //Value used to pass the exlusive aggregate from lookback
  
[numthreads(GROUP_SIZE, 1, 1)]
void CD_Simple(int3 gtid : SV_GroupThreadID)
{
    int partitionIndex;
    do
    {
        if (gtid.x == 0)
            InterlockedAdd(b_state[PARTITIONS], 1, g_sharedMem[0]);
        GroupMemoryBarrierWithGroupSync();
        partitionIndex = WaveReadLaneFirst(g_sharedMem[0]);
        GroupMemoryBarrierWithGroupSync();
```
  
The algorithm begins by atomically incrementing the index value in the `b_state` buffer. Using `InterlockedAdd`, the pre-increment value is read back into shared memory. We then broadcast the value from shared memory into thread registers. As we have mentioned in the overview, this prevents deadlocking issues.
<br>
<br>
## Reduce
  
![Chained 2](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/68788f9b-01cb-4796-ad9b-e0fb515f67ec)

```HLSL
      g_sharedMem[LANE + WAVE_PART_START] = b_prefixSum[LANE + PARTITION_START];
      g_sharedMem[LANE + WAVE_PART_START] += WavePrefixSum(g_sharedMem[LANE + WAVE_PART_START]);

      for (int i = LANE + WAVE_PART_START + LANE_COUNT; i < WAVE_PART_END; i += LANE_COUNT)
      {
          g_sharedMem[i] = b_prefixSum[i + PARTITION_START];
          g_sharedMem[i] += WavePrefixSum(g_sharedMem[i]) + WaveReadLaneFirst(g_sharedMem[i - 1]);
      }
      GroupMemoryBarrierWithGroupSync();

      if (gtid.x < WAVES_PER_GROUP)
          g_sharedMem[SPINE_INDEX] += WavePrefixSum(g_sharedMem[SPINE_INDEX]);
      GroupMemoryBarrierWithGroupSync();
```
  
Once index of the partition tile has been acquired, we load its elements into shared memory. Because the elements will remain resident in shared memory we perform the initial inclusive prefix scans now, producing the partition tile aggregate value in the process.
<br>
<br>
## LookBack
![Aggregates](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/6836fa80-e324-413e-a882-0c8b9f6e4db0)

```HLSL
      if (gtid.x == 0)
      {
          if (partitionIndex == 0)
              InterlockedOr(b_state[partitionIndex], FLAG_INCLUSIVE ^ (g_sharedMem[PARTITION_MASK] << 2));
          else
              InterlockedOr(b_state[partitionIndex], FLAG_AGGREGATE ^ (g_sharedMem[PARTITION_MASK] << 2));
          g_breaker = true;
      }
```
 
We now atomically post the partition tile aggregate into device memory (**labelled block aggregate in the above figure**), and update the flag value to `1` indicating that the aggregate of that partition tile is available. If the partition tile index is `0`, that is to say if it is the first tile, we 
update the flag value to `2`, indicating that the inclusive sum of all preceeding tiles is included in this value. This is useful because it signals to other threadblocks during the *look back* that they can stop upon reaching this value, because again, the sum of all preceeding tiles is included in this value. To elide use of barriers to maintain coherency between the aggregate value and the flag value, we bit pack them into the same value, allowing us to maintain coherency with atomics only. However, as our flag value takes up 2 bits, this limits the total aggregate sum that can be calculated to $2^{30}$.
#

![Lookback](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/207b0f97-22fa-4bc6-94dd-d286ff33d544)

```HLSL
      uint aggregate = 0;
      if (partitionIndex)
      {
          if (gtid.x < LANE_COUNT)
          {
              for (int k = partitionIndex - gtid.x - 1; 0 <= k;)
              {
                  uint flagPayload = b_state[k];
                  const int inclusiveIndex = WaveActiveMin(LANE + LANE_COUNT - ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE ? LANE_COUNT : 0));
                  const int gapIndex = WaveActiveMin(LANE + LANE_COUNT - ((flagPayload & FLAG_MASK) == FLAG_NOT_READY ? LANE_COUNT : 0));
                  if (inclusiveIndex < gapIndex)
                  {
                      aggregate += WaveActiveSum(LANE <= inclusiveIndex ? (flagPayload >> 2) : 0);
                      if (LANE == 0)
                      {
                          InterlockedAdd(b_state[partitionIndex], 1 | aggregate << 2);
                          g_aggregate = aggregate;
                      }
                      break;
                  }
                  else
                  {
                      if (gapIndex < LANE_COUNT)
                      {
                          aggregate += WaveActiveSum(LANE < gapIndex ? (flagPayload >> 2) : 0);
                          k -= gapIndex;
                      }
                      else
                      {
                          aggregate += WaveActiveSum(flagPayload >> 2);
                          k -= LANE_COUNT;
                      }
                  }
              }
          }
          GroupMemoryBarrierWithGroupSync();
      
          //propogate aggregate values
          if (WAVE_INDEX || LANE)
              aggregate = WaveReadLaneFirst(g_aggregate);
          GroupMemoryBarrierWithGroupSync();
      }
```
  
Once the aggregate has been posted, we enter the *look back* phase. First, we restrict the lookback to the first warp of the threadblock, as the full paralellism is not required and the overhead of coordinating the full threadblock is more expensive than it is worth. Each thread of the warp reads a the combined flag and aggregate value of the partition tiles directly preceeding its own tile which we call `flagPayload`. We then use `flagPayload` along with the warp-wide minimum function to calculate two values `inclusiveIndex` and `gapIndex`. These respectively indicate whether an inclusive aggregate has been found in this run of values, and whether a partition tile that has posted no sum (a "gap") has been found. If an `inclusiveIndex` has been found which does not have a `gapIndex` blocking it, the sum of the preceeding tiles has been found and the value is atomically posted back into device memory. If no `inclusiveIndex` has been found, or if it is blocked by a `gapIndex`, we sum all the aggregates up to the `gapIndex` and shift our lookback window up to the `gapIndex` then `break` and repeat the process.    

The first partition tile skips this phase, because it has no preceeding sums.
  
## Propagation
  
![propagation](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/bce170cd-b105-42f9-a926-f582da88e571)
  
```HLSL
      const uint prev = (WAVE_INDEX ? WaveReadLaneFirst(g_sharedMem[WAVE_PART_START - 1]) : 0) + aggregate;
      for (int i = LANE + WAVE_PART_START; i < WAVE_PART_END; i += LANE_COUNT)
          b_prefixSum[i + PARTITION_START] = g_sharedMem[i] + (i < WAVE_PART_END - 1 ? prev : aggregate);
        
  } while (partitionIndex + THREAD_BLOCKS < PARTITIONS);
```
Once the preceeding tile aggregate has been found, the value is broadcast to the other warps in the group, and the prefix sum is completed.
                                                        
</details>

## The Deadlock Issue
First, let us assume that there are more partition tiles than the maximum number of threadblocks which can be hosted on the GPU.  Given that there are $\frac{inputSize}{partitionTileSize}$ partition tiles, and we use a partition tile of size 8192, this is almost always true.

Now let's take a look at the *lookback* phase of the algorithm:
```HLSL
uint aggregate = 0;
if (partitionIndex)
{
    if (gtid.x < LANE_COUNT)
    {
        for (int k = partitionIndex - gtid.x - 1; 0 <= k;)
        {
            uint flagPayload = b_state[k];
            const int inclusiveIndex = WaveActiveMin(LANE + LANE_COUNT - ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE ? LANE_COUNT : 0));
            const int gapIndex = WaveActiveMin(LANE + LANE_COUNT - ((flagPayload & FLAG_MASK) == FLAG_NOT_READY ? LANE_COUNT : 0));
            if (inclusiveIndex < gapIndex)
            {
                aggregate += WaveActiveSum(LANE <= inclusiveIndex ? (flagPayload >> 2) : 0);
                if (LANE == 0)
                {
                    InterlockedAdd(b_state[partitionIndex], 1 | aggregate << 2);
                    g_aggregate = aggregate;
                }
                break;
            }
            else
            {
                if (gapIndex < LANE_COUNT)
                {
                    aggregate += WaveActiveSum(LANE < gapIndex ? (flagPayload >> 2) : 0);
                    k -= gapIndex;
                }
                else
                {
                    aggregate += WaveActiveSum(flagPayload >> 2);
                    k -= LANE_COUNT;
                }
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    //propogate aggregate values
    if (WAVE_INDEX || LANE)
        aggregate = WaveReadLaneFirst(g_aggregate);
    GroupMemoryBarrierWithGroupSync();
}
```

Each partition tile is serially dependent on the reduced sum of *all* preceding partition tiles. During the *lookback* phase we essentially create a threadblock level spin-lock that spins until all preceding sums have been posted in `b_state`. However, recall that GPU is free to schedule the execution of threadblocks in ***any order it wants.*** So for example, for 100 scheduled threadblocks a possible execution order is: threadblock 100, threadblock 51, threadblock 67 . . . Furthermore, in programming environments like HLSL or Metal, threadblocks follow the ([***Occupancy Bound Execution paradigm***](https://drops.dagstuhl.de/opus/volltexte/2018/9561/pdf/LIPIcs-CONCUR-2018-23.pdf)). Basically, once a threadblock begins executing on an SM, it ***must run to completion.*** 

So here is the problem: Say we have 100 partition tiles, but our GPU can host a maximum of 32 threadblocks. If we use the `SV_GroupID` to assign partition tiles, e.g. threadblock 0 processes partition tile 0, threadblock 1 processes partition tile 1 and so on, there is a chance that the GPU is completely occupied by threadblocks that each have a preceding partition tile that has not been processed or begun being processed. ***Because our spin-lock only unlocks when all preceding sums are posted, this results in a deadlock. Because of the occupancy bound execution model, the GPU can't switch to a new threadblock if one is stalled.*** This crashes the kernel.

So going back to our example, imagine we begin execution of our algorithm and our GPU begins executing threadblocks 10 to 41 which each begin processing their corresponding partition tile. However, they're each dependent on the partition tiles 0 to 9, and upon entering the lookback phase begin waiting for those sums to be posted. Because of the occupancy bound execution model, the GPU can't switch execution to threadblocks 0 to 9, and so the threadblocks spin until the kernel crashes.

The solution to this is quite clever. Instead of assigning partition tiles by `SV_GroupID`, we initialize a single index in device memory to `0`. This is `b_index`. Upon beginning execution, each threadblock atomically fetches and increments `b_index`, like so:
```HLSL
  if (gtid.x == 0)
      InterlockedAdd(b_state[PARTITIONS], 1, g_sharedMem[0]);
  GroupMemoryBarrierWithGroupSync();
  partitionIndex = WaveReadLaneFirst(g_sharedMem[0]);
  GroupMemoryBarrierWithGroupSync();
```

The value fetched from `b_index` is the partition tile that will be processed, and incrementing `b_index` by 1 signals that the tile has begun being processed. So for example:

+ `b_index` is initialized to 0.

+ threadblock 99 begins execution, fetches 0 and increments by 1. Begins processing partition tile 0.

+ threadblock 26 begins execution, fetches 1 and increments by 1. Begins processing partition tile 1.

+ threadblock 53 begins execution, fetches 2 and increments by 1. Begins processing partition tile 2.

By assigning partition tiles this way, ***you guarantee that no threadblock begins execution of a partition tile with preceding partition tiles that have not been processed or begun being processed.*** And because of that, you guarantee that the deadlock cannot occur. I believe this solves the forward progress portability issue discussed in Levien's [blog](https://raphlinus.github.io/gpu/2020/04/30/prefix-sum.html) post.


# Testing Methodology
Performance testing in the Unity HLSL environment is challenging because, to the best of my knowledge, there is no way to directly time the execution of kernels in situ on GPU's. Instead we make do by making an [`AsyncGPUReadback.Request`](https://docs.unity3d.com/ScriptReference/Rendering.AsyncGPUReadback.html), which essentially waits until the GPU signals that is finished working on a buffer, then reads the entire buffer back from GPU memory to CPU memory. Typically a GPU to CPU readback operation [introduces adds a large amount of latency to the timing](https://therealmjp.github.io/posts/gpu-memory-pool/), so we create a single element dummy buffer, `b_timing`, whose small size renders this readback latency irrelevant. 

Although this does accurately time the execution of the kernel, there is another issue: Unity limits the size of buffers to 2^28 elements, which experimental testing of a simple `MemCpy` kernel indicates is insufficient to fully utilize the memory bandwidth. Running the following kernel at `k = 1` (a single iteration of the inner loop) and an input size of 2^28, we achieve a harmonic mean performance of ~37.8 G 32-bit elements/sec. Although it is well known that graphics cards rarely achieve the speed listed by the manufacturer, given that our 2080 Super has a listed memory bandwidth of 496 GB/sec, this is far short of the theoretical maximum `MemCpy` speed of ~62 G 32-bit elements/sec. 
<br>
<br>
```HLSL
extern int e_size;
extern int e_repeats;
RWBuffer<uint4> bufferA;
RWBuffer<uint4> bufferB;
RWBuffer<uint> timingBuffer;

[numthreads(GROUP_SIZE, 1, 1)]
void MemCpy(int3 id : SV_DispatchThreadID)
{
    const uint inc = GROUP_SIZE * THREAD_BLOCKS;
    const uint vecSize = e_size >> 2;
    const uint repeats = e_repeats;
    
    for (int k = 0; k < repeats; ++k)
    {
        for (int i = id.x; i < vecSize; i += inc)
            bufferA[i] = bufferB[i];
    }
    
    //to time the kernel execution
    if (id.x == 0)
        timingBuffer[id.x] = id.x;
}
```
<br>

We can confirm that the input size *is* affecting the speed of the `MemCpy` by increasing `k`, effectively running the `MemCpy` at input sizes of `k` * 2^28. Indeed, running 1000 tests at each `k` from 1 to 40, we get the following result:
<br>
<br>

![MemCpy Timing at k Loop Repititions](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/82ba7705-39de-4705-9c08-8b4468d10c8a)
<br>
<br>

This reveals two things:

+ At larger `k`,  the `MemCpy` reaches speeds of up to ~48.8 G 32-bit ele/sec, which is further confirmed by the slope of the best fit line, which indicates a speed of 49.1 G 32-bit ele/sec. 

+ As indicated by the intercept of our best fit line, there appears to be some residual latency introduced by the CPU side timing.
## 

Because *Chained Scan with Decoupled Lookback* has the sorting-network-like quality that it performs the same amount of work regardless of the input, we can use the same looping approach as we did with `MemCpy`. Testing our *Chained Scan with Decoupled Lookback* implementation, we get an almost identical result, reaching a harmonic mean performance of ~38.3 G 32-bit ele/sec at `k = 1`. Performing the same testing as we did with the `MemCpy` we get the following result:
<br>
<br>

![Chained Scan With Decoupled Lookback Speed at k Loop Repititions](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/3443466f-4a24-4801-b11d-414abb6af5f7)

Showing that our implementation achieves a speed of ~54.1 G 32-bit ele/sec, or a memory bandwidth utilization of about ~87%. This falls in line almost perfectly with the results achieved in the original paper, where a bandwidth utilization of ~85% was achieved on an Nvidia Tesla M40. Although our result is seemingly higher than the `MemCpy`, this is almost certainly due to the fact that our *Chained Scan with Decoupled Lookback* implementation reads and writes several runs of elements at a time, instead of reading and immediately writing as in our `MemCpy`, thus allowing the SM's to better hide memory latency.

To test the timings on your own computer, I have included the timing version of *Chained Scan with Decoupled Lookback* as well as the `MemCpy` kernel in the folder `TimingPrefixSums`.
# Interesting Reading and Bibliography

Duane Merrill and Michael Garland. “Single-pass Parallel Prefix Scan with De-coupled Lookback”. In: 2016. 
url: https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

Grimshaw, Andrew S. and Duane Merrill. “Parallel Scan for Stream Architectures.” (2012).
url: https://libraopen.lib.virginia.edu/downloads/6t053g00z

Matt Pettineo. GPU Memory Pools in D3D12. Jul. 2022.
url: https://therealmjp.github.io/posts/gpu-memory-pool/

Ralph Levien. Prefix sum on portable compute shaders. Nov. 2021. 
url: https://raphlinus.github.io/.

Tyler Sorensen, Hugues Evrard, and Alastair F. Donaldson. “GPU Schedulers: How Fair Is Fair Enoughl”. In: 29th International Conference on Concurrency Theory (CONCUR 2018). Ed. by Sven Schewe and Lijun Zhang. Vol. 118. Leibniz International Proceedings in Informatics (LIPIcs). Dagstuhl, Germany: Schloss Dagstuhl–Leibniz-Zentrum fuer Informatik, 2018, 23:1–23:17. isbn: 978-3-95977-087-3. doi: 10.4230/LIPIcs.CONCUR.2018.23. 
url: http://drops.dagstuhl.de/opus/volltexte/2018/9561.

Vasily Volkov. “Understanding Latency Hiding on GPUs”. PhD thesis. EECS Department, University of California, Berkeley, Aug. 2016. 
url: http://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.html

Zhe Jia et al. Dissecting the NVidia Turing T4 GPU via Microbenchmarking. 2019. arXiv: 1903.07486.
url: https://arxiv.org/abs/1903.07486

