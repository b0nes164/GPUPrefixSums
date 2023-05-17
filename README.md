# GPU Prefix Sums
![Prefix Sum Speeds, in Unity Editor, RTX 2080 Super](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/7fd486be-cfd5-4a03-b24b-2d850431d8fd)

This project is a survey of GPU prefix sums, ranging from the warp to the device level, with the aim of providing developers an uncompiled look at modern prefix sum implementations. In particular, this project was inspired by Duane Merill's [research](https://research.nvidia.com/person/duane-merrill%2520iii), and it includes various implementations of Merill and Garland's [Chained Scan with Decoupled Lookback](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back), which is how we are able to reach speeds approaching `MemCopy()`. While this project is not intended to be the fastest possible implementation, it is still highly optimized and is still significantly (5x) faster than a naive implementation of Blelloch's [algorithm](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda) on a threadblock level. Finally, this project was written in HLSL for compute shaders, though with reasonable knowledge of GPU programming it is easily portable. **To the best of my knowledge, all algorithms included in this project are in the public domain and free to use, as is this project itself (Chained Scan is licensed under BSD-2, and Blelloch's algorithm was released through GPU Gems).** 

# Important Notes
<details>
  <summary>This project has NOT been tested on AMD video cards or on CPU integrated graphics. If you have an AMD card, preprocessor macros for wave/warp size MUST be MANUALLY CHANGED in the desired scan file.</summary>
&nbsp;
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Unfortunately, AMD and Nvidia video cards have different wave sizes, which means that code that uses wave intrinsic functions, like we do, must be manually tuned for each video card brand. So if you have an AMD card, you will have to manaully change the preprocessor macros in the .compute file. To do so, open up the `.compute` file of the desired scan. Inside you will find the preprocessor macros like so:

  ![image](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/a1290a27-4106-4b3e-81d9-26a2e41bcca6)
&nbsp;  

  Comment out or delete the Nvidia values, and uncomment the AMD values. However, to reiterate, these scans have not been tested on AMD hardware, and should be treated as such.
&nbsp;   
  
&nbsp;   
</details>

<details>
  <summary>Chained Scan with Decoupled Lookback is not guaranteed to work on Nvidia cards older than Volta or AMD cards older than ????.</summary>
&nbsp;  
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Chained Scan relies on two concepts to function properly: guaranteed forward progress of threads and fair scheduling of thread groups. This is because we are effectively creating threadblock level spinlocks to alleviate the serial dependency of threadblocks in chained scan. Without these guaruntees, there is a chance that a threadblock never unlocks or that a threadblock whose depedent aggregate is already available is kept waiting for a suboptimal period of time. Thus hardware models without these features may not see the same speedup or may fail to work altogether. If you wish to read more about the portability issues, and some of the general challenges of implementing chained decoupled scan, I would highly recommend reading Raph Levien’s [blog](https://raphlinus.github.io/gpu/2020/04/30/prefix-sum.html) detailing his experience with it. To read more on the issue of GPU workgroup progress models I recommend this [paper](https://arxiv.org/abs/2109.06132).
&nbsp;  
  
&nbsp;  
</details>

<details>
  <summary>Currently the maximum aggregate sum supported in Chained Scan with Decoupled Lookback is 2^30.</summary>
&nbsp;  
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In order to maintain coherency of the flag values between threadblocks, we have to pack the threadblock aggregate into into the same value as the status flag. As the flag takes up two bits, we are left 30 bits for the aggregate. Although shader model 6.6 does support 64-bit values and atomics, enabling these features in Unity is difficult, and I have chosen not include it until Unity releases the features in earnest.
&nbsp;  

&nbsp; 
</details>

<details>
  <summary>DX12 is a must as well as a minimum Unity version of 2021.1 or later</summary>
&nbsp;  
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;As we make heavy use of [WaveIntrinsics](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/hlsl-shader-model-6-0-features-for-direct3d-12), we need `pragma use_dxc` [to access shader model 6.0](https://forum.unity.com/threads/unity-is-adding-a-new-dxc-hlsl-compiler-backend-option.1086272/).
&nbsp;  
  
</details>

# To Use This Project
To use Chained Scan, simply attatch `ChainedDecoupledScan.cs` to a game object, then attach the compute shader which matches 'serialized field' on the game object. This is a bit clunky, but as the group sizes and other macro defined values are designed to be able to vary between the different Chained Scan implementations, this is the simplest way I've come up with (**If you know a way to vary the threadgroup size at compile time, please let me know, that would be awesome**).

To use any of the block-level scans or below, simply attatch 'PrefixSumDispatcher.cs' to a game object, then attach 'PrefixSums.compute' to the script.
 
# Kogge-Stone
![KoggesStoneImage](https://user-images.githubusercontent.com/68340554/224911618-6f54231c-251f-4321-93ec-b244a0af49f7.png)

# Sklansky
![SklanskyFinal](https://user-images.githubusercontent.com/68340554/224912079-b1580955-b702-45f9-887a-7c1003825bf9.png)

# Brent-Kung-Blelloch
![BrentKungImage](https://user-images.githubusercontent.com/68340554/224912128-73301be2-0bba-4146-8e20-2f1f3bc7c549.png)

# Reduce-Then-Scan
![ReduceScanFinal](https://user-images.githubusercontent.com/68340554/224912530-2e1f2851-f531-4271-8246-d13983ccb584.png)

# Radix Brent-Kung-Blelloch
![RadixBrentKungImage](https://user-images.githubusercontent.com/68340554/224912635-88550d08-f2c2-4c97-b8a2-8fcebc939d41.png)

# Radix Sklansky
![RadixSklanskyFinal](https://user-images.githubusercontent.com/68340554/224912704-97d6eacf-9f33-4ac1-ab12-ad89e92cec51.png)

# Radix Reduce
![RadixReduceImage](https://user-images.githubusercontent.com/68340554/224912791-5fa3743e-df00-49e7-8d37-028b73bba211.png)
