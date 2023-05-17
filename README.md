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
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In order to maintain coherency of the flag values between threadblocks, we have to pack the threadblock aggregate into into the same value as the status flag. As the flag takes up 2 bits, we are left 30 bits for the aggregate. Although shader model 6.6 does support 64-bit values and atomics, enabling these features in Unity is difficult, and I have chosen to not include it until Unity moves the feature out of beta.
&nbsp;  

&nbsp; 
</details>

<details>
  <summary>DX12 is a must as well as a minimum Unity version of 2021.1 or later</summary>
&nbsp;  
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;As we make heavy use of [WaveIntrinsics](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/hlsl-shader-model-6-0-features-for-direct3d-12), we need `pragma use_dxc` [to access shader model 6.0](https://forum.unity.com/threads/unity-is-adding-a-new-dxc-hlsl-compiler-backend-option.1086272/).
&nbsp;  
  
&nbsp; 
</details>

<details>
  <summary>All scans are inclusive.</summary>
</details>

# To Use This Project
1. Download or clone the repository.
2. Drag the contents of `src` into a desired folder within a Unity project.
3. Attach the scan dispatcher to an empty game object. All scan dispatchers are named  `ScanNameHere + Dispatcher.cs`.
4. Attach the matching compute shader to the game object. All compute shader are named `ScanNameHere.compute`. The dispatcher will return an error if you attach the wrong shader.

If you did this correctly you should see this in the inspector:



![image](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/70bb5097-fff2-44e5-b396-2930a059fbad)

Ensure the sliders are set to actual values, and you are good to go!
 
 # Testing Suite
 ![Tests](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/9d79a090-2f13-4031-925a-ef0788e75bf3)

Every scan dispatcher inherits a testing suite that can be controlled in the inspector.

+ `Validate Sum` performs a prefix sum on a buffer of size 2^`SizeExponent`. If you have `Validate Text` ticked, any errors found will printed in the debug log. For very large sums, this can take several minutes, so if you don't want absolutely every error printed you can tick `Quick Text` which limits the number of errors printed to 1024.

+ `Validate Sum Random`/ `Validate Sum Monotonic` are tests that fill the buffer with either random numbers or a monotonically increasing sequence of numbers. To make debugging easier, the buffer is by default filled with the value 1. This makes error checking very simple, because the corresponding correct prefix sum is the positive integers up to the size of the buffer. However, this makes some other errors possible, so to cover our bases we include this test.

+ `Debug At Size` performs a prefix sum on a buffer of size `Specific Size`. This is a way to directly test the validty of the prefix sum on buffer sizes that are not powers of two.

+ `Debug State` prints the contents of the `State Buffer` into the debug log after dispatching a scan. The `State Buffer` contains the threadblock aggregates in the device level scans, so `Debug State` can be used to verify that the aggregation is performed correctly.

+ `Torture Test` performs `Kernel Iterations` number of scans at the current `Size Exponent`. It does not perform any validation and should be used to verify the stability of a scan.  

# Kogge-Stone
![KoggesStoneImage](https://user-images.githubusercontent.com/68340554/224911618-6f54231c-251f-4321-93ec-b244a0af49f7.png)

# Sklansky
![SklanskyFinal](https://user-images.githubusercontent.com/68340554/224912079-b1580955-b702-45f9-887a-7c1003825bf9.png)

# Brent-Kung-Blelloch
![BrentKungImage](https://user-images.githubusercontent.com/68340554/224912128-73301be2-0bba-4146-8e20-2f1f3bc7c549.png)

# Reduce-Then-Scan
![ReduceScanFinal](https://user-images.githubusercontent.com/68340554/224912530-2e1f2851-f531-4271-8246-d13983ccb584.png)

# Raking Reduce-Scan
![RakingReduceScan](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/3bc46762-1d61-41aa-aee5-1c492ff76ad6)

# Warp-Sized Radix Brent-Kung-Blelloch
![WarpBrentKung](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/671768cf-a536-42bd-bd46-ddd6695c75e6)

# Warp-Sized Radix Sklansky
![WarpSklansky](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/e7dd3c56-334f-431a-a7bf-0d30fa64ea9a)

# Warp-Sized Radix Reduce-Scan
![WarpRakingReduce](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/72997c9e-ae94-41f0-83e8-c1122530f2e4)

#Block Level Scans
![Block Level 1](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/306d0908-da29-45a2-9f79-fea4c3856560)
![Block Level 2](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/d5d47250-f5ff-4156-85cd-6ba2639fa237)
