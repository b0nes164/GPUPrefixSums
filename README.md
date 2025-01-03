# GPU Prefix Sums

![Prefix Sum Roundup](https://github.com/user-attachments/assets/a0ecca1d-a18f-4a1d-b8d2-8db130c4be83)

GPUPrefixSums aims to bring state-of-the-art GPU prefix sum techniques from CUDA and make them available in portable compute shaders. In addition to this, it contributes "Decoupled Fallback," a novel fallback technique for Chained Scan with Decoupled Lookback that should allow devices without forward thread progress guarantees to perform the scan without crashing. The D3D12 implementation includes an extensive survey of GPU prefix sums, ranging from the warp to the device level; all included algorithms utilize wave/warp/subgroup (referred to as "wave" hereon) level parallelism but are completely agnostic of wave size. As a measure of the quality of the code, GPUPrefixSums has also been implemented in CUDA and benchmarked against Nvidia's [CUB](https://github.com/NVIDIA/cccl) library. Although GPUPrefixSums aims to be portable to any wave size supported by HLSL, [4, 128], due to hardware limitations, it has only been tested on wave sizes 4, 16, 32, and 64. You have been warned!

If you are interested in prefix sums for their use in radix sorting, check out GPUPrefixSum's sibling repository [GPUSorting](https://github.com/b0nes164/GPUSorting)!

# Decoupled Fallback

In Decoupled Fallback, a threadblock will spin for a set amount of cycles while waiting for the reduction of a preceding partition tile. If the maximum spin count is exceeded, the threadblock is free to perform a fallback operation. Multiple thread blocks are allowed to perform fallbacks on the same deadlocking tile, but through use of atomic compare and swap, only one thread block ends up broadcasting its reduction in device memory. Although this means potentially performing redundant calculations, the upside is that fallback performance is no longer limited by the latency of signal propagation between thread blocks.

As of writing this 9/22/2024, Decoupled Fallback shows promising results on Apple M GPU's. However the version included here are out of date, with the most up-to-date development occuring in [Vello](https://github.com/linebender/vello).

# Survey

![Prefix](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/de5504c4-42a9-494f-b707-cbdf66c93cc9)

A prefix sum, also called a scan, is a running total of a sequence of numbers at the n-th element. If the prefix sum is inclusive the n-th element is included in that total, if it is exclusive, the n-th element is not included. The prefix sum is one of the most important algorithmic primitives in parallel computing, underpinning everything from [sorting](https://arxiv.org/abs/2206.01784), to [compression](https://arxiv.org/abs/1311.2540), to [graph traversal](https://dl.acm.org/doi/10.1145/2370036.2145832).

# Basic Scans

<details open>
<summary>

### Kogge-Stone
  
</summary>

![KoggesStoneImage](https://user-images.githubusercontent.com/68340554/224911618-6f54231c-251f-4321-93ec-b244a0af49f7.png)
  
</details>

<details>

<summary>

### Sklansky

</summary>

![SklanskyFinal](https://user-images.githubusercontent.com/68340554/224912079-b1580955-b702-45f9-887a-7c1003825bf9.png)

</details>

<details>

<summary>

### Brent-Kung

</summary>

![BrentKungImage](https://user-images.githubusercontent.com/68340554/224912128-73301be2-0bba-4146-8e20-2f1f3bc7c549.png)

</details>

<details>

<summary>

### Reduce Scan

</summary>

![ReduceScanFinal](https://user-images.githubusercontent.com/68340554/224912530-2e1f2851-f531-4271-8246-d13983ccb584.png)

</details>

<details>

<summary>

### Raking Reduce-Scan

</summary>

![RakingReduceScan](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/3bc46762-1d61-41aa-aee5-1c492ff76ad6)

</details>

# Warp-Synchronized Scans
  
<details open>

<summary>

### Warp-Sized-Radix Brent-Kung
  
</summary>

![RadixBrentKung](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/d246240a-c088-4a74-a4c9-3cc2de12d1c9)

</details>
  
<details>

<summary>

### Warp-Sized-Radix Brent-Kung with Fused Upsweep-Downsweep

</summary>

![RadixBKFused](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/4ccfaf59-9864-4405-a89f-f950c70cda2b)

</details>
  
<details>

<summary>

### Warp-Sized-Radix Sklansky

</summary>

![WarpSklansky](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/e7dd3c56-334f-431a-a7bf-0d30fa64ea9a)

</details>

<details>

<summary>

### Warp-Sized-Radix Serial

</summary>

![RadixSerial](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/11dccbcb-7ab4-4a99-aacb-704d0c4581fd)

</details>
  
<details>

<summary>

### Warp-Sized-Radix Raking Reduce-Scan

</summary>

![WarpRakingReduce](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/72997c9e-ae94-41f0-83e8-c1122530f2e4)

</details>

# Block-Level Scan Pattern

<details>

<summary>

### First Partition

</summary>

![Block Level 1](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/306d0908-da29-45a2-9f79-fea4c3856560)

</details>

<details>

<summary>

### Second Partition and Onwards

</summary>

![Block Level 2](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/d5d47250-f5ff-4156-85cd-6ba2639fa237)

</details>

# Device Level Scan Pattern (Reduce-Then-Scan)

<details>

<summary>

### Reduce

</summary>

![Device 1](https://github.com/user-attachments/assets/81c76371-5378-4554-bf0d-c6e6de2465de)

</details>

<details>

<summary>

### Scan Along the Intermediate Reductions

</summary>

![Device 2](https://github.com/user-attachments/assets/87712018-0f7a-401d-b81f-08e2ba4515a3)


</details>

<details>

<summary>

### Scan and Pass in Intermediate Values

</summary>

![Device 3](https://github.com/user-attachments/assets/0bc7a8f4-dc67-4932-80f6-61aa2eda7236)

</details>

# Getting Started

## GPUPrefixSumsD3D12

Headless implementation in D3D12, includes:
* Reduce then Scan
* Chained Scan with Decoupled Lookback
* Chained Scan with Decoupled Lookback Decoupled Fallback

Requirements:
* Visual Studio 2019 or greater
* Windows SDK 10.0.20348.0 or greater

The repository folder contains a Visual Studio 2019 project and solution file. Upon building the solution, NuGet will download and link the following external dependencies:
* [DirectX 12 Agility SDK](https://www.nuget.org/packages/Microsoft.Direct3D.D3D12)
* [DirectX Shader Compiler](https://www.nuget.org/packages/Microsoft.Direct3D.DXC/1.8.2403.18)
* [Microsoft Windows Implementation Library](https://www.nuget.org/packages/Microsoft.Windows.ImplementationLibrary/)
  
See the repository wiki for information on running tests.

## GPUPrefixSumsCUDA

GPUPrefixSumsCUDA includes:
* Reduce then Scan
* Chained Scan with Decoupled Lookback

The purpose of this implementation is to benchmark the algorithms and demystify their implementation in the CUDA environment. It is not intended for production or use; instead, a proper implementation can be found in the CUB library.

Requirements:
* Visual Studio 2019 or greater
* Windows SDK 10.0.20348.0 or greater
* CUDA Toolkit 12.3.2
* Nvidia Graphics Card with Compute Capability 7.x or greater.

The repository folder contains a Visual Studio 2019 project and solution file; there are no external dependencies besides the CUDA toolkit. The use of sync primitives necessitates Compute Capability 7.x or greater. See the repository wiki for information on running tests.

## GPUPrefixSumsUnity

Released as a Unity package includes:
* Reduce then Scan
* Chained Scan with Decoupled Lookback

Requirements:
* Unity 2021.3.35f1 or greater

Within the Unity package manager, add a package from git URL and enter:

`https://github.com/b0nes164/GPUPrefixSums.git?path=/GPUPrefixSumsUnity`

See the repository wiki for information on running tests.

## GPUPrefixSumsWGPU

### WARNING: TESTING ONLY CURRENTLY, NOT FULLY PORTABLE

Barebones implementation--no vectorization, no wave intrinsics--to be used as a testbed.

Requirements:
* wgpu 22.0
* pollster 0.3
* bytemuck 1.16.3

# Interesting Reading and Bibliography

Duane Merrill and Michael Garland. “Single-pass Parallel Prefix Scan with De-coupled Lookback”. In: 2016. 
url: https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

Grimshaw, Andrew S. and Duane Merrill. “Parallel Scan for Stream Architectures.” (2012).
url: https://libraopen.lib.virginia.edu/downloads/6t053g00z

Matt Pettineo. GPU Memory Pools in D3D12. Jul. 2022.
url: https://therealmjp.github.io/posts/gpu-memory-pool/

Ralph Levien. Prefix sum on portable compute shaders. Nov. 2021. 
url: https://raphlinus.github.io/gpu/2021/11/17/prefix-sum-portable.html

Tyler Sorensen, Hugues Evrard, and Alastair F. Donaldson. “GPU Schedulers: How Fair Is Fair Enoughl”. In: 29th International Conference on Concurrency Theory (CONCUR 2018). Ed. by Sven Schewe and Lijun Zhang. Vol. 118. Leibniz International Proceedings in Informatics (LIPIcs). Dagstuhl, Germany: Schloss Dagstuhl–Leibniz-Zentrum fuer Informatik, 2018, 23:1–23:17. isbn: 978-3-95977-087-3. doi: 10.4230/LIPIcs.CONCUR.2018.23. 
url: http://drops.dagstuhl.de/opus/volltexte/2018/9561.

Vasily Volkov. “Understanding Latency Hiding on GPUs”. PhD thesis. EECS Department, University of California, Berkeley, Aug. 2016. 
url: http://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.html

Zhe Jia et al. Dissecting the NVidia Turing T4 GPU via Microbenchmarking. 2019. arXiv: 1903.07486.
url: https://arxiv.org/abs/1903.07486
