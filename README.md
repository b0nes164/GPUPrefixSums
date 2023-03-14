# GPU Prefix Sums

This project is a survey of various prefix sums, ranging from the warp to the device level. In particular it includes a compute shader implementation of Merill and Garland's [Chained Scan with Decoupled Lookback](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back). To the best of my knowledge, all algorithms included in this project are in the public domain and free to use, as is this project itself(Chained Scan is licensed under BSD-2, and Blelloch's algorithm was released through GPU Gems). 

# Important Notes
<details>
  <summary>Currently the maximum aggregate sum supported in the Chained Scan algorithm is 2^30.</summary>  <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This is because in order to maintain globlal coherency of the flag values between threadblocks/workgroups, we have to pack the group aggregate into into the same value as the group status flag which takes up 2 bits. Although shader model 6.6 does support 64-bit values and atomics, these features are not available in Unity compute shaders due to a bug, I believe.
 <br/>
</details>

<details>
  <summary>Chained Scan is not guaranteed to work on AMD cards or on Nvidia cards older than Volta.</summary>   <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Because Chained Scan relies on the guaranteed forward progress of threads and fair scheduling of thread groups, I cannot guarantee that this implementation will work on AMD cards or on Nvidia cards older than Volta. This is because unlike CPUs, GPUs are far less standardized and [different hardware models have vastly different capabilities](https://arxiv.org/abs/2109.06132). Therefore, this code is more of a proof of concept, rather than something that I would recommend implementing into a production build (eventually I will update this project to include a device level reduce-then scan which is a tad slower but more than suffecient, and more importantly does not have the hardware portability issues that Chained Scan does). If you wish to read more about the portability issues, and some of the general challenges of implementing Chained scan, I would highly recommend reading Raph Levien’s [blog](https://raphlinus.github.io/gpu/2020/04/30/prefix-sum.html) detailing his experience with it.
 <br/>
</details>


<details>
  <summary>DX12 is a must as well as a minimum Unity version of 2021.1 or later</summary>   <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; As we make heavy use of [WaveIntrinsics](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/hlsl-shader-model-6-0-features-for-direct3d-12), we need `pragma use_dxc` [to access shader model 6.0](https://forum.unity.com/threads/unity-is-adding-a-new-dxc-hlsl-compiler-backend-option.1086272/).
 <br/>
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
