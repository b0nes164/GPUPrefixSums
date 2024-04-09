# GPU Prefix Sums
![Prefix Sum Speed Comparison](https://github.com/b0nes164/GPUPrefixSums/assets/68340554/f2088c9d-a3e5-4b6c-b318-4765501f9fb6)

# NOTICE THIS PROJECT IS UNDERGOING INTENSIVE REVISIONS, AND IS LIABLE TO CHANGE FREQUENTLY. The Unity portion is outdated and contains known bugs. D3D12 and CUDA should are up to date. The survey and readme are being rewritten and have been temporarily removed.

GPUPrefixSums aims to bring state-of-the-art GPU prefix sum techniques from CUDA and make them available in portable compute shaders. It includes an extensive survey of GPU prefix sums, ranging from the warp to the device level, and in particular it includes implementations of Merill and Garland's [Chained Scan with Decoupled Lookback](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back), which is how we are able to reach speeds approaching `MemCopy()`. Finally, this project was written in HLSL for compute shaders, though with reasonable knowledge of GPU programming it is easily portable. 

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