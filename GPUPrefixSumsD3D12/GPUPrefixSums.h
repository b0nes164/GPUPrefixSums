/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 12/2/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
#pragma once
#include "pch.h"

namespace GPUPrefixSums {
    struct DeviceInfo {
        std::wstring Description;
        std::wstring SupportedShaderModel;
        uint32_t SIMDWidth;
        uint32_t SIMDLaneCount;
        uint32_t SIMDMaxWidth;
        uint64_t dedicatedVideoMemory;
        uint64_t sharedSystemMemory;
        bool SupportsWaveIntrinsics;
        bool SupportsReduceThenScan;
        bool SupportsChainedScanDecoupledLookback;
    };
}  // namespace GPUPrefixSums