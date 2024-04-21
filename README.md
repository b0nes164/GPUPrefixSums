# GPU Prefix Sums

![GPUPrefixSums vs CUB Roundup](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/ec2ce84d-a06a-4d7b-969d-52d851025220)

GPUPrefixSums aims to bring state-of-the-art GPU prefix sum techniques from CUDA and make them available in portable compute shaders. In addition to this, it contributes "Decoupled Fallback," a novel fallback technique for Chained Scan with Decoupled Lookback that should allow devices without forward thread progress guarantees to perform the scan without crashing. The D3D12 implementation includes an extensive survey of GPU prefix sums, ranging from the warp to the device level; all included algorithms utilize wave/warp/subgroup (referred to as "wave" hereon) level parallelism but are completely agnostic of wave size. As a measure of the quality of the code, GPUPrefixSums has also been implemented in CUDA and benchmarked against Nvidia's [CUB](https://github.com/NVIDIA/cccl) library. Although GPUPrefixSums aims to be portable to any wave size supported by HLSL, [4, 128], due to hardware limitations, it has only been tested on wave sizes 4, 16, 32, and 64. You have been warned!

If you are interested in prefix sums for their use in radix sorting, check out GPUPrefixSum's sibling repository [GPUSorting](https://github.com/b0nes164/GPUSorting)!

# Decoupled Fallback

In Decoupled Fallback, a threadblock will spin for a set amount of cycles while waiting for the reduction of a preceding partition tile. If the maximum spin count is exceeded, the threadblock is free to perform a fallback operation. Multiple thread blocks are allowed to perform fallbacks on the same deadlocking tile, but through use of atomic compare and swap, only one thread block ends up broadcasting its reduction in device memory. Although this means potentially performing redundant calculations, the upside is that fallback performance is no longer limited by the latency of signal propagation between thread blocks.

As of writing this (4/19/2024), it is unclear whether this method poses any advantages over the "scalar fallback" method first proprosed by Levien [here](https://raphlinus.github.io/gpu/2021/11/17/prefix-sum-portable.html). More testing is required.

# Survey

![Prefix](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/de5504c4-42a9-494f-b707-cbdf66c93cc9)

A prefix sum, also called a scan, is a running total of a sequence of numbers at the n-th element. If the prefix sum is inclusive the n-th element is included in that total, if it is exclusive, the n-th element is not included. The prefix sum is one of the most important algorithmic primitives in parallel computing, underpinning everything from [sorting](https://arxiv.org/abs/2206.01784), to [compression](https://arxiv.org/abs/1311.2540), to [graph traversal](https://dl.acm.org/doi/10.1145/2370036.2145832).

The survey can be found in the wiki.

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
