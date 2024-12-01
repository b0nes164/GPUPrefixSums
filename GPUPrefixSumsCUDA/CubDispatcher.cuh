/******************************************************************************
 * CUB Implementations
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Based off of Research by:
 *          Duane Merrill, Nvidia Corporation
 *          Michael Garland, Nvidia Corporation
 *          https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back
 *
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 ******************************************************************************/
#pragma once
#include "UtilityKernels.cuh"
#include "cub/device/device_scan.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CubDispatcher {
    const uint32_t k_maxSize;
    uint32_t* m_scan;

   public:
    CubDispatcher(uint32_t size) : k_maxSize(size) {
        cudaMalloc(&m_scan, k_maxSize * sizeof(uint32_t));
    }

    ~CubDispatcher() { cudaFree(m_scan); }

    void BatchTimingCubChainedScan(uint32_t size, uint32_t batchCount) {
        if (size > k_maxSize) {
            printf("Error, requested test size exceeds max initialized size. \n");
            return;
        }

        printf("Beginning CUB ChainedScanDecoupledLookback inclusive batch timing test at:\n");
        printf("Size: %u\n", size);
        printf("Test size: %u\n", batchCount);

        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, m_scan, size);

        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float totalTime = 0.0f;
        for (uint32_t i = 0; i <= batchCount; ++i) {
            InitOne<<<256, 256>>>(m_scan, size);
            cudaDeviceSynchronize();
            cudaEventRecord(start);
            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, m_scan, size);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float millis;
            cudaEventElapsedTime(&millis, start, stop);
            if (i)
                totalTime += millis;

            if ((i & 15) == 0)
                printf(". ");
        }

        printf("\n");
        totalTime /= 1000.0f;
        printf("Total time elapsed: %f\n", totalTime);
        printf("Estimated speed at %u 32-bit elements: %E keys/sec\n\n", size,
               size / totalTime * batchCount);
    }
};