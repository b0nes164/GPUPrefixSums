# GPU Prefix Sums

This project is a survey of various prefix sums, ranging from the warp to the device level. In particular it includes a compute shader implementation of Merill and Garland's [Chained Scan with Decoupled Lookback](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back). To the best of my knowledge, all algorithms included in this project are in the public domain and free to use, as is this project itself(Chained Scan is licensed under BSD-2, and Blelloch's algorithm was released through GPU Gems). 

#Important Notes
Currently, shader model 6.6 features such as 64-bit values and atomics are not available in Unity compute shaders due to a bug, I believe. Because Chained Scan relies on the global coherency of flag values, I cannot guaruntee my current implementations of the algorithm will run correctly *all* of the time, as the only is to pack the flag values.

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
